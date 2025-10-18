#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run a remote LLM API to get responses."""
import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import pathlib
from copy import deepcopy
from openai import OpenAI

from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
# random.seed(0) # 随机化对于API调用可能不是必需的，除非您仍想打乱文档顺序

# --- API 调用函数 ---
# 将API调用封装成一个函数，更清晰，也便于错误处理和重试
def get_api_response(client, model_name, prompt, temperature, top_p, max_new_tokens):
    """
    Sends a request to the API and returns the response.
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please answer the user's question based on the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens, # 注意：OpenAI API 使用 max_tokens
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API call failed with error: {e}")
        return f"Error: API call failed. Details: {e}"


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    api_key,
    base_url,
    rate_limit,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    max_new_tokens,
    output_path,
):
    # --- 1. 初始化 API 客户端 ---
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api-key argument or DASHSCOPE_API_KEY environment variable.")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    logger.info(f"API client initialized for model '{model_name}' at endpoint '{base_url}'")


    # --- 2. 加载和准备 Prompts (这部分逻辑保持不变) ---
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    examples = []
    prompts = []
    all_model_documents = []
    
    with xopen(input_path) as fin:
        for line in tqdm(fin, desc="Preparing prompts"):
            input_example = json.loads(line)
            question = input_example["question"]
            
            if closedbook:
                documents = []
            else:
                documents = [Document.from_dict(ctx) for ctx in deepcopy(input_example["ctxs"])]
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # 随机化逻辑保持不变
                import random
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )
            
            # Instruct 格式化可能不再需要，因为API模型通常默认就是对话/指令格式
            # 如果需要，可以保留 format_instruct_prompt 函数并使用
            # prompt = format_instruct_prompt(prompt)

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)


    # --- 3. 循环调用 API 获取回复 (核心修改部分) ---
    responses = []
    # 不再需要 torch, GPU, 或 batching
    for prompt in tqdm(prompts, desc="Getting API responses"):
        response = get_api_response(
            client=client,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        responses.append(response)
        
        # 速率控制，避免超出API频率限制
        time.sleep(1 / rate_limit)


    # --- 4. 写入输出文件 (这部分逻辑保持不变) ---
    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


# 辅助函数，原样保留
def format_instruct_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    PROMPT_FOR_GENERATION = f"{INTRO_BLURB}\n{INSTRUCTION_KEY}\n{instruction}\n{RESPONSE_KEY}\n"
    return PROMPT_FOR_GENERATION


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    
    # --- 必填参数 ---
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    
    # --- API 相关参数 ---
    parser.add_argument(
        "--model",
        help="Model name to use for the API call",
        required=True,
        default="qwen2.5-7b-instruct"
    )
    parser.add_argument(
        "--api-key",
        help="API Key for the service. If not provided, tries to read from DASHSCOPE_API_KEY environment variable.",
        default=None
    )
    parser.add_argument(
        "--base-url",
        help="The base URL for the API.",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    parser.add_argument(
        "--rate-limit",
        help="Requests per second to limit API calls.",
        type=float,
        default=2.0 # 默认每秒2次请求
    )

    # --- 生成控制参数 (与API兼容) ---
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )

    # --- Prompt 构建参数 (保留) ---
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )

    args = parser.parse_args()

    # 移除了所有与GPU和本地模型batching相关的参数
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.api_key,
        args.base_url,
        args.rate_limit,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
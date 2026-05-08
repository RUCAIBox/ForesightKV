import argparse
import json
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3

dataset2key = {
    "gpqa": ["input", "output"],
}

dataset2max_length = {
    "gpqa": 32768,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def process_prompt(question):
    chat_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Please solve the following problem step by step and put the final "
                    "answer within \\boxed{}.\n" + question
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat_prompt


def _last_token_id(tokenizer, text, fallback_id):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids:
        return ids[-1]
    return fallback_id


def _validate_tokenizer(tokenizer, tokenizer_source, model_path):
    vocab_size = len(tokenizer)
    if vocab_size < 1000:
        raise ValueError(
            f"Loaded tokenizer from `{tokenizer_source}` has suspicious vocab size={vocab_size}. "
            "This usually means the checkpoint directory does not contain a real tokenizer. "
            f"Please provide `--tokenizer_path` explicitly, model_path={model_path}."
        )


def main(args):
    with open(args.save_path, "w") as fout:
        prompts = []
        test_data = []

        with open(args.dataset_path) as f:
            for index, line in enumerate(f):
                example = json.loads(line)
                question_key, answer_key = dataset2key["gpqa"]
                if question_key not in example or answer_key not in example:
                    continue

                question = example[question_key]
                for _ in range(args.times):
                    copied = {
                        question_key: example[question_key],
                        answer_key: example[answer_key],
                        "question": question,
                        "index": index,
                    }
                    prompt = process_prompt(question)
                    copied["prompt"] = prompt
                    prompts.append(prompt)
                    test_data.append(copied)

        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            batch_prompts = prompts[i : i + args.eval_batch_size]
            tokenized_prompts = tokenizer(
                batch_prompts,
                padding="longest",
                return_tensors="pt",
                add_special_tokens=True,
            ).to("cuda")

            prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

            output = model.generate(
                **tokenized_prompts,
                max_length=args.max_length,
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
                num_beams=1,
            )

            batch_token_stats = []
            for j in range(output.size(0)):
                total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())
                prefill = prefill_lengths[j]
                output_tokens = total_tokens - prefill
                batch_token_stats.append(
                    {
                        "sample_idx": i + j,
                        "prefill_tokens": prefill,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            batch_outputs = tokenizer.batch_decode(
                [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
                skip_special_tokens=True,
            )

            torch.cuda.empty_cache()

            for j, generation in enumerate(batch_outputs):
                sample_idx = batch_token_stats[j]["sample_idx"]
                test_data[sample_idx]["prompt"] = batch_prompts[j]
                test_data[sample_idx]["generation"] = generation
                test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
                test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
                test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
                test_data[sample_idx]["sample_idx"] = sample_idx
                fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer source path. If set, use this tokenizer instead of model_path.",
    )
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    parser.add_argument(
        "--method",
        type=str,
        default="fullkv",
        choices=[
            "rkv",
            "fullkv",
            "snapkv",
            "streamingllm",
            "h2o",
            "foresightkv",
            "foresightkv_topk",
        ],
    )
    parser.add_argument("--kv_budget", type=int, default=None)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", type=bool, default=True)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument(
        "--divide_method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument("--divide_length", type=int, default=256)
    parser.add_argument(
        "--compression_content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    seed = int(time.time())
    print(seed)
    set_seed(seed)

    args.dataset_name = "gpqa"
    if args.max_length == -1:
        args.max_length = dataset2max_length[args.dataset_name]

    compression_config = {
        "method": args.method,
        "method_config": {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
        },
        "compression": None,
        "update_kv": args.update_kv,
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
    }

    tokenizer_source = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, use_fast=True, padding_side="left"
    )
    _validate_tokenizer(tokenizer, tokenizer_source, args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.method.lower() != "fullkv":
        hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model_type = getattr(hf_config, "model_type", "").lower()
        model_path_lower = args.model_path.lower()

        if "llama" in model_path_lower or model_type == "llama":
            replace_llama(compression_config)
        elif "qwen3" in model_path_lower or model_type == "qwen3":
            replace_qwen3(compression_config)
        elif "qwen" in model_path_lower or model_type == "qwen2":
            replace_qwen2(compression_config)
        else:
            raise ValueError(
                f"Unsupported model: {args.model_path} (model_type={model_type})"
            )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    model.config.update(model_config)

    if args.method.lower() != "fullkv":
        fallback_id = tokenizer.eos_token_id
        if fallback_id is None:
            fallback_id = tokenizer.pad_token_id
        if fallback_id is None:
            fallback_id = 0
        model.newline_token_ids = [
            _last_token_id(tokenizer, "\n", fallback_id),
            _last_token_id(tokenizer, ".\n", fallback_id),
            _last_token_id(tokenizer, ")\n", fallback_id),
            _last_token_id(tokenizer, "\n\n", fallback_id),
            _last_token_id(tokenizer, ".\n\n", fallback_id),
            _last_token_id(tokenizer, ")\n\n", fallback_id),
        ]
        model.after_think_token_ids = [
            _last_token_id(tokenizer, "</think>", fallback_id),
        ]

    main(args)

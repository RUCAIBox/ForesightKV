import argparse
import csv
import json
import os

from tqdm import tqdm

from data_loader import load_data_vanilla
from evaluate import evaluate
from parser import choice_answer_clean, parse_ground_truth, run_execute
from python_executor import PythonExecutor
from utils import save_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="gpqa-eval", type=str)
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument(
        "--base_dir",
        default="./results",
        type=str,
        help="Folder containing JSON/JSONL files to be evaluated",
    )
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument(
        "--stop_words",
        default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："],
        type=list,
    )
    parser.add_argument("--dataset", default="gpqa", type=str)
    return parser.parse_args()


def prepare_data(data_name, args):
    examples = load_data_vanilla(args.input_path)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{args.exp_name}/{data_name}", exist_ok=True)

    processed_samples = []
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(data_name, args):
    examples, processed_samples = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for cnt, example in tqdm(enumerate(examples), total=len(examples)):
        example["solution"] = example.get("solution", example.get("output"))
        idx = example.get("idx", cnt)
        example["question"] = example.get("question", example.get("input", ""))

        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }

        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "output",
            "domain",
            "difficulty",
            "source",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    codes = []
    for example in examples:
        code = example["generation"]
        for stop_word in args.stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i]
        result = results[i]
        preds = [result[0]]
        reports = [result[1]]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code)
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    out_dir = os.path.join(args.output_dir, args.exp_name, data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"{getattr(args, 'size', 'default')}-{getattr(args, 'method', 'default')}_math_eval.jsonl",
    )
    save_jsonl(all_samples, out_file)

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json


def main_all(args):
    json_files = []
    for file in os.listdir(args.base_dir):
        filepath = os.path.join(args.base_dir, file)
        if os.path.isfile(filepath) and (file.endswith(".json") or file.endswith(".jsonl")):
            json_files.append(filepath)

    if not json_files:
        print("No JSON/JSONL files found in the folder.")
        return

    results_table = {}
    for json_file in json_files:
        dataset = args.dataset
        args.input_path = json_file
        args.size = "default"
        args.method = dataset
        print(f"Processing: dataset={dataset} from file {json_file}")
        result_json = main(dataset, args)
        results_table[json_file] = result_json.get("acc", None)

    output_csv = os.path.join(args.output_dir, "all_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Accuracy"])
        for dataset, acc in sorted(results_table.items()):
            writer.writerow([dataset, acc])
    print(f"All evaluation results have been saved to {output_csv}")


if __name__ == "__main__":
    args = parse_args()
    main_all(args)

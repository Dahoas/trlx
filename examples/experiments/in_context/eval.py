import torch
from util import load_jsonl, calculator, append_jsonl, dump_jsonl
from data import MaskedSFTDataset
from accelerate import Accelerator
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import re
import random
from datasets import load_dataset
from logger import Logger
from typing import Optional
import json
import numpy as np
import sympy


def clean_answer(answer : str):
    answer = answer.split("A: ")[-1]
    # Remove all but first finall answer (given after ####)
    if len(answer.split("####")) < 2:
        return answer
    trace, final_answer = answer.split("####")[:2]
    # Take rhs of any final equals
    final_answer = final_answer.split("=")[-1]
    # Remove units (with preceding white space) from final answer
    final_answer = re.sub(r"\s(?:[a-zA-Z\s]|(?:\*\*\d+))+/?(?:[a-zA-Z\s]|(?:\*\*\d+))*", "", final_answer)
    # Remove spaces, commas, and newlines
    final_answer = final_answer.replace(" ", "")
    final_answer = final_answer.replace(",", "")
    final_answer = final_answer.replace("\n", "")
    final_answer = final_answer.replace("$", "")
    return trace + "####" + final_answer

def to_decimal(expr):
    try:
        expr = str(float(sympy.simplify(expr)))
        # If int only print int
        if round(expr) == expr:
            expr = round(expr)
    except:
        expr = expr
    return expr


def compare_final_answers(model_final_answer, gt_final_answer):
    try:
        return int(float(sympy.simplify(model_final_answer)) == float(sympy.simplify(gt_final_answer)))
    except:
        return int(model_final_answer == gt_final_answer)

def eval_answer(model_answer : str, gt_answer : str, mode : str = "sparse", rubric : Optional[str] = None, rm : Optional[PreTrainedModel] = None) -> int:
    model_answer = clean_answer(model_answer)
    gt_answer = clean_answer(gt_answer)
    model_final_answer = model_answer.split("####")[-1]
    gt_final_answer = gt_answer.split("####")[-1]
    if mode == "sparse":
        return compare_final_answers(model_final_answer, gt_final_answer), model_final_answer, gt_final_answer
    elif mode == "heuristic":
        # Simply check for correct sequence of computations according to gt
        gt_answer_trace = re.findall(r"<<(.*)>>", gt_answer)
        gt_answer_trace = set([to_decimal(s.split("=")[-1]) for s in gt_answer_trace])
        model_answer_trace = re.findall(r"<<(.*)>>", model_answer)
        model_answer_trace = set([to_decimal(s.split("=")[-1]) for s in model_answer_trace])
        # +1 for correct compution, total max of -1 for any irrelevant computation
        # Cheats a little bit by assuming numbers are unique
        shared = len(gt_answer_trace.intersection(model_answer_trace))
        gt_extra = len(gt_answer_trace - model_answer_trace)
        model_extra = len(model_answer_trace - gt_answer_trace)
        return (shared + max(0, min(1, model_extra))) / (len(gt_answer_trace) + 1), model_final_answer, gt_final_answer
    elif mode == "model rubric":
        assert rubric is not None
        raise NotImplementedError
    elif mode == "rm":
        assert rm is not None
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported evaluation model {mode}!")


def eval_problem(model_final_answers, model_scores, gt_final_answer, mode="max"):
    if mode == "max":
        return max(model_scores)
    elif mode == "voting":
        counts = {}
        maj_ans = model_final_answers[0]
        for mfa in model_final_answers:
            counts[mfa] = 0 if counts.get(mfa) is None else 1 + counts.get(mfa)
            maj_ans = maj_ans if counts[mfa] < counts[maj_ans] else mfa
        return compare_final_answers(maj_ans, gt_final_answer)
    else:
        raise NotImplementedError


def score_stats(scores):
    scores = torch.tensor(scores, dtype=torch.float32)
    stats = {}
    stats["mean"] = torch.mean(scores).item()
    stats["max"] = torch.max(scores).item()
    stats["min"] = torch.min(scores).item()
    stats["std"] = torch.std(scores).item()
    stats["num"] = len(scores)
    return stats

def assign_score_label(score, min_score=0, max_score=1, num_labels=3):
    # Assigned by rounding down
    if score >= max_score:
        return score
    labels = np.linspace(start=min_score, stop=max_score, num=num_labels)
    ind = np.argmax(labels > score)
    return labels[ind-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--logging_path", default=None, type=str)
    parser.add_argument("--mode", type=str, default="sparse", help="Type of evaluation to do")
    parser.add_argument("--model_path", type=str, default="/private/home/alexdahoas/repos/ckpts/vicuna_13B")
    parser.add_argument("--tok_path", type=str, default="oobabooga/llama-tokenizer")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--num_labels", default=3, type=int, help="Number of labels to be used for score conditioning")
    parser.add_argument("--bon_scheme", default="max", type=str, help="Scheme for deciding final answer from multiple samples")
    parser.add_argument("--split", default="test", type=str)
    args = parser.parse_args()

    Logger.init(args.logging_path)

    dataset = load_jsonl(args.eval_dataset) if ".jsonl" in args.eval_dataset else load_dataset(args.eval_dataset)[args.split]
    if args.mode == "sparse":
        for sample in tqdm(dataset):
            score, mfa, gfa = eval_answer(sample["model_response"], sample["answer"], mode=args.mode)
            sample.update({"mfa": mfa, "gfa": gfa, "score": score, "score_label": score})
    elif args.mode == "heuristic":
        for sample in tqdm(dataset):
            score, mfa, gfa = eval_answer(sample["model_response"], sample["answer"], mode=args.mode)
            sample["score"] = score
            sample["score_label"] = assign_score_label(score, num_labels=args.num_labels)
            sample.update({"mfa": mfa, "gfa": gfa})
    else:
        raise NotImplementedError
    if args.logging_path is not None:
        dump_jsonl(args.logging_path, dataset)

    # Collect score stats
    sample_stats = score_stats([sample["score"] for sample in dataset])
    # Group by label
    labels = np.linspace(0, 1, args.num_labels)
    label_stats = {label: {"num": 0} for label in labels}
    for sample in dataset:
        label_stats[sample["score_label"]]["num"] += 1
    # Group by prompt
    prompt_scores = {}
    for sample in dataset:
        sample_prompt = sample["prompt"]
        pre_score = prompt_scores.get(sample_prompt)
        if pre_score is None:
            prompt_scores[sample_prompt] = {"mfas": [sample["mfa"]], "scores": [sample["score"]], "gta": sample["gfa"], "K": 1}
        else:
            prompt_scores[sample_prompt]["scores"].append(sample["score"])
            prompt_scores[sample_prompt]["mfas"].append(sample["mfa"])
            prompt_scores[sample_prompt]["K"] += 1
    for res in prompt_scores.values():
        res.update({"score": eval_problem(res["mfas"], res["scores"], res["gta"], mode=args.bon_scheme)})
    prompt_stats = score_stats([v["score"] for v in prompt_scores.values()])
    prompt_stats["K"] = prompt_scores[sample_prompt]["K"]
    prompt_stats["scheme"] = args.bon_scheme
    save_stats = {"dataset": args.eval_dataset, "mode": args.mode, "sample_stats": sample_stats, "prompt_stats": prompt_stats, "label_stats": label_stats}
    print(json.dumps(save_stats))
    append_jsonl("logs/master_results.jsonl", save_stats)


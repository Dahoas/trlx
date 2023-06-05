from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from util import load_jsonl, dump_jsonl, parallelize
import multiprocessing as mp
import random
import json


# Only predicts on response tokens
# TODO(dahoas): fix this at some point
class MaskedSFTDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.random_clause_dropout = .2
            self.agglomeration_level = 0
            self.num_omit = 8
            self.data = data
            self.tokenizer = tokenizer
            self.EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]
            self.preprocess(data, tokenizer)

        def preprocess(self, data, tokenizer):
            self.SEP = ""#" ;"
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            self.prompts = []
            self.responses = []
            max_length = max([len(tokenizer.encode(ele["prompt"] + self.SEP + ele["response"] + '<|endoftext|>')) for ele in tqdm(data)])
            self.max_length = max_length
            print("Max length: {}".format(max_length))

            # Data expected in prompt response pairs
            for ele in tqdm(data):
                prompt, response = ele["prompt"], ele["response"]
                prompt_encoding_len = len(tokenizer(prompt)["input_ids"])
                encodings_dict = tokenizer(prompt + self.SEP + response + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length")
                input_id = torch.tensor(encodings_dict['input_ids'])
                attn_mask = torch.tensor(encodings_dict['attention_mask'])
                label_mask = (input_id == self.EOS_ID).type(torch.int32)
                first_eos = label_mask.nonzero()
                # Skip text which has no eos token
                if len(first_eos) == 0:
                    continue
                else:
                    first_eos = first_eos[0, 0]
                label_mask[first_eos] = 0  # Want to predict on first eos_token
                assert(prompt_encoding_len >= 1)
                label_mask[:prompt_encoding_len] = 1  # Do not predict on prompt
                flipped_mask = 1 - label_mask
                self.input_ids.append(input_id)
                self.attn_masks.append(attn_mask)
                self.labels.append(self.input_ids[-1] * flipped_mask - 100 * label_mask)
                self.prompts.append(prompt)
                self.responses.append(response)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx], self.labels[idx], self.prompts[idx], self.responses[idx]

        def split_data(self, txt):
            return txt.splitlines()

def l2d(l):
    return {k: [s.get(k) for s in l] for k in l[0]}

def format():
    gsm8k = load_dataset("gsm8k", "socratic")
    gsm8k = gsm8k.map(lambda s: {"prompt": "Q: " + s["question"]+'\nA: ', "response": s["answer"]})
    test = gsm8k["test"]
    train = HFDataset.from_dict(gsm8k["train"][:-256])
    val = HFDataset.from_dict(gsm8k["train"][-256:])
    gsm8k = DatasetDict({"train": train, "val": val, "test": test})
    gsm8k.push_to_hub("Dahoas/cot_gsm8k_socratic")
    return gsm8k

def merge_by_prompt(sample, d2):
    """Combine sample with rows from d2 sharing same prompt
    """
    d1_prompt = sample["prompt"]
    for d2_sample in d2:
        d2_prompt = d2_sample["prompt"]
        if d1_prompt == d2_prompt:
            for k, v in d2_sample.items():
                sample[k] = v
            break
    return sample

def format_conditional_training(sample):
    sample["prompt"] = sample["prompt"] + "[Score: {}] ".format(sample["score_label"])
    return sample

def inspect_dataset(data_path, index, split="train", keys=None):
    dataset = load_jsonl(data_path) if ".jsonl" in data_path else load_dataset(data_path)[split]
    sample = dataset[index]
    keys = list(sample.keys()) if keys is None else keys
    print(json.dumps({k: sample[k] for k in keys}, indent=2))

def add_label(sample):
    sample["score_label"] = 1.0
    return sample

def stack_datasets():
    clean_dataset_total = load_dataset("Dahoas/cot_gsm8k_socratic").map(add_label)
    clean_dataset = list(clean_dataset_total["train"])
    synthetic_dataset = load_jsonl("logs/merged_vicuna_7B_answers_heuristic_score.jsonl")
    #clean_dataset = format_conditional_training(clean_dataset)
    #synthetic_dataset = format_conditional_training(synthetic_dataset)
    dataset = clean_dataset + synthetic_dataset
    random.shuffle(dataset)
    dump_jsonl("logs/socratic_conditional_finetune.jsonl", dataset)
    train = HFDataset.from_dict(l2d(dataset)).remove_columns(["score", "model_response"])
    val = clean_dataset_total["val"]
    test = clean_dataset_total["test"]
    dataset = DatasetDict({"train": train, "val": val, "test": test}).map(format_conditional_training)
    dataset.push_to_hub("Dahoas/gsm_socratic_conditional")
    #inspect_dataset("logs/socratic_conditional_finetune.jsonl", 0, keys=["prompt", "response"])


if __name__ == "__main__":
    inferences = load_jsonl("logs/vicuna_7B_answers.jsonl")
    for sample in inferences:
        sample["model_response"] = sample.pop("response")
    gsm = load_dataset("Dahoas/cot_gsm8k")["train"]
    # gsm = load_dataset("Dahoas/cot_gsm8k").map(add_label).map(format_conditional_training)["train"]
    merged = parallelize(merge_by_prompt, inferences, {"d2": gsm}, num_procs=48)
    print(merged[0])
    dump_jsonl("logs/merged_vicuna_7B_answers.jsonl", merged)

    
    
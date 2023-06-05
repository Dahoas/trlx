import torch
from util import load_jsonl, calculator
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
import sympy

EOS_TOK = ""
EOS_TOK_ID = 2
BOS_TOK_ID = 1
DENOM_TOK_ID = 6778
ANSWER_TOK_ID = 4136

# custom stopping criteria STOPS when last line has a call( * )
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tok = None):
        self.tok = tok
        return
    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        toks = list(input_ids[0])
        last_tok = toks[-1]
        return DENOM_TOK_ID == last_tok or EOS_TOK_ID == last_tok or (ANSWER_TOK_ID == last_tok and ANSWER_TOK_ID in toks[:-1])
        #text_output = self.tok.decode(toks)
        #last_line_res = re.findall(r"<<([^\s]*=[^\s]*)>>", text_output)
        #return len(last_line_res) > 0 or EOS_TOK in text_output

stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria()])

#TODO: 1. Stop generation when </call> is made to speed up eval.
###### 2. Sync subcalled generation across all gpus with an allreduce. Necessary for parallel eval. Currently only working with batch size = 1
def subcall_model(model, tok, tok_prompts, max_new_tokens, temp):
    """Routine allowing for dynamic calls to calculator tool. 
    Implemented only for batch size = 1
    """
    global stopping_criteria
    CALC_LIMIT = 20
    calc_calls = 0
    assert len(tok_prompts) == 1
    while calc_calls < CALC_LIMIT:
        output = model.generate(tok_prompts, max_new_tokens=max_new_tokens, do_sample=True, temperature=temp, stopping_criteria=stopping_criteria)
        text_output = tok.batch_decode(output)
        # Break if terminator found
        if "####" in text_output[0]:
            break
        elif ">>" in text_output[0]:
            text_output = text_output[0]
            # Find last calculator call
            try:
                args = re.findall(r"<<(.*)>>", text_output)[-1]
                # Take lhs of equals
                args = args.split("=")[0]
                ret = calculator(args)
                # Put args and ret back together
                res = "<<{}={}>>".format(args, ret)
                next_prompt = "<<".join(text_output.split("<<")[:-1]) + res
            except:
                next_prompt = text_output
            tok_prompts = tok(next_prompt, return_tensors="pt").input_ids.to(tok_prompts.device)
            # Remove extra bos ids
            if tok_prompts[0][0].item() == BOS_TOK_ID:
                tok_prompts = tok_prompts[:, 1:]
            # Increment calls to calc
            calc_calls += 1
        else:
            break
    return text_output


def infer(model, dataloader, tokenizer, max_new_tokens, temp, K):
    """Function to infer causal model in parallel on dataloader 
    with at most max_length tokens at temperature temp.
    """
    scores = []
    for inputs in tqdm(dataloader):
        # TODO(dahoas): Log everything. Change model response keys to model_response
        prompts, responses = inputs["prompts"], inputs["responses"]
        tok_prompts = tokenizer(prompts, return_tensors="pt").input_ids.to(accelerator.device)
        for _ in range(K):
            text_outputs = subcall_model(model, tokenizer, tok_prompts, max_new_tokens, temp)
            model_answers = [s.split(EOS_TOK)[0] for s in text_outputs]
            outputs = [{"prompt": p, "response": r} for p, r in zip(prompts, model_answers)]
            Logger.log(outputs)


if __name__ == "__main__":
    # Command to run: accelerate launch --num_procs 1 infer.py 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset", type=str)
    parser.add_argument("--logging_path", type=str)
    parser.add_argument("--model_path", type=str, default="/private/home/alexdahoas/repos/ckpts/vicuna_13B")
    parser.add_argument("--tok_path", type=str, default="oobabooga/llama-tokenizer")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--K", default=1, type=int, help="Rollouts per prompt")
    parser.add_argument("--eight_bit", action="store_true")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    Logger.init(args.logging_path)

    tok = AutoTokenizer.from_pretrained(args.tok_path)
    tok.pad_token = tok.eos_token
    EOS_TOK = tok.eos_token
    EOS_TOK_ID = tok(EOS_TOK).input_ids[-1]
    BOS_TOK_ID = tok(tok.bos_token).input_ids[0]
    # NOTE: This is assuming >> gets tokenized together as in Llama, (also assuming #### gets tokenized together)
    DENOM_TOK_ID = tok("3>>").input_ids[-1]
    ANSWER_TOK_ID = tok("\n#### 3").input_ids[-3]
    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(tok)])

    dataset = load_jsonl(args.prompt_dataset) if ".jsonl" in args.prompt_dataset else load_dataset(args.prompt_dataset)[args.split]
    dataset = MaskedSFTDataset(dataset, tok)

    prompt_dataset = dataset
    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=True, device_map="auto") if args.eight_bit else AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()
    if not args.eight_bit:
        print("Moving to half precision...")
        model.half()

    data_collator = lambda data: {
                                    'prompts': [f[3] for f in data], 
                                    'responses': [f[4] for f in data]
                                 }
    dataloader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    model = accelerator.unwrap_model(model)
    model = model.to(accelerator.device)

    infer(model, dataloader, tok, args.max_new_tokens, args.temp, args.K)
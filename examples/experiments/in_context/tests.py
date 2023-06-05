import json
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sympy
from util import calculator
from eval import eval_answer, clean_answer


#####Tests#####

def test_bitsnbytes():
    model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=True, device_map="auto")
    tok = AutoTokenizer.from_pretrained("gpt2")
    m_out = model.generate(tok("hello", return_tensors="pt").input_ids, max_new_tokens=60)


def test_calculator():
    inp = "1+2-3*4/5"
    out = calculator("1+2-3*4/5")
    print("Input: ", inp, "Out: ", out)
    assert out == "0.6"


def test_clean_answer():
    inp = "<s> Q: Between them, Mark and Sarah have 24 traffic tickets. Mark has twice as many parking tickets as Sarah, and they each have an equal number of speeding tickets. If Sarah has 6 speeding tickets, how many parking tickets does Mark have?\nA:  How many parking tickets does Mark have? ** Mark has 6 / 2 = <<6/2=3>>3 parking tickets\nHow many parking tickets does Mark have? ** He has 3 + 6 = <<3+6=9>>9 parking tickets.\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets\n#### 9 parking tickets"
    out = clean_answer(inp)
    print("Input:")
    print(inp)
    print("Output:")
    print(out)
    assert out == """ How many parking tickets does Mark have? ** Mark has 6 / 2 = <<6/2=3>>3 parking tickets
How many parking tickets does Mark have? ** He has 3 + 6 = <<3+6=9>>9 parking tickets.
####9"""


def run_tests():
    test_calculator()
    test_clean_answer()


if __name__ == "__main__":
     run_tests()
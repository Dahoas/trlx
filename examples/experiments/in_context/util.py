import json
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sympy
import multiprocessing as mp


#####Data util#####

def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data

def dump_jsonl(filename, data):
    with open(filename, "w") as f:
        for dict_t in data:
                json.dump(dict_t, f)
                f.write("\n")

def append_jsonl(filename, data):
    with open(filename, "a+") as f:
        json.dump(data, f)
        f.write("\n")

def parallelized_func(func, data, rank, save_dict, kwargs):
    save_dict[rank] = [func(sample, **kwargs) for sample in tqdm(data)]
    return save_dict

def parallelize(func, data, kwargs, num_procs=24):
    batch_size = (len(data) + num_procs - 1) // num_procs
    batches = [data[i * batch_size : (i+1) * batch_size] for i in range(num_procs)]
    procs = []
    manager = mp.Manager()
    save_dict = manager.dict()
    for rank, batch in enumerate(batches):
        p = mp.Process(target=parallelized_func, args=(func, batch, rank, save_dict, kwargs))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    # Flatten the result
    res = [s for sl in save_dict.values() for s in sl]
    return res

#####Tools####

def calculator(expr : str) -> str:
    """Implements calculator using sympy. Assumes expression is in the form x . y . z where . is one of four basic arithmetic ops.
    Returns final answer as a decimal.
    """
    expr = sympy.simplify(expr)
    # If int make int
    if round(float(expr)) == float(expr):
        expr = int(expr)
    return str(expr)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from prompts import *


device = "cuda:1"

def infer(model, tok, prompt):
    m_in = tok(prompt, return_tensors="pt").input_ids
    m_in = m_in.to(device)
    m_out = model.generate(m_in, max_new_tokens=128, do_sample=True, temperature=0.7)
    out = tok.batch_decode(m_out)
    return out

if __name__ == "__main__":
    gsm8k = load_dataset("gsm8k", "socratic")["train"]
    problem_1 = gsm8k["question"][0]
    answer_1 = gsm8k["answer"][0]
    problem_2 = gsm8k["question"][2]
    answer_2 = gsm8k["answer"][2]
    prompt = few_shot_rubric_prompt.format(problem_1, rubric_1, student_solution_1, rubric_feedback_1, problem_2, rubric_2, student_solution_2)
    prompt = few_shot_rubric_design_prompt.format(problem_1, answer_1, rubric_1, problem_2, answer_2)
    prompt = few_shot_simple_refinement_prompt.format(student_solution_1, critique_feedback_1, refinement_1, student_solution_2, critique_feedback_2)
    prompt = few_shot_sentence_refinement_prompt.format(student_solution_1, critique_feedback_1, refinement_1, student_solution_2, critique_feedback_2)

    model = AutoModelForCausalLM.from_pretrained("/private/home/alexdahoas/repos/ckpts/vicuna_13B")
    model = model.half()
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")
    
    out = infer(model, tok, prompt)
    print(out)



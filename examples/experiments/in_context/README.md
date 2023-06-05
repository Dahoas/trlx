# Learning to reason via critique conditional reinforcment learning

Folder for reasoning critique project

## Resources

- Project doc: [here](https://docs.google.com/document/d/1XUYjH60UUlz97ot6Q9CtoEKHbD7pMcEXFgnKEUYreqM/edit?usp=sharing)

# Scripts

## Running Slurm

- srun --gpus-per-node=8 --partition=learnlab --time=24:00:00 --cpus-per-task 48 --constraint volta32gb --exclusive --pty /bin/bash -l

## Train

- accelerate launch --num_processes 8 --config_file configs/train_z2.yaml gsm8k_sft.py
- accelerate launch --num_processes 8 --config_file configs/train_z3.yaml gsm8k_sft.py
- accelerate launch --num_processes 8 --config_file configs/train_z2.yaml gsm8k_ppo.py

## Infer

- accelerate launch --num_processes 8 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/cot_gsm8k --logging_path logs/vicuna_13B_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 128 --model_path /private/home/alexdahoas/repos/ckpts/vicuna_13B --eight_bit

- CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/cot_gsm8k --logging_path logs/vicuna_13B_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 128 --model_path /private/home/alexdahoas/repos/ckpts/vicuna_13B --eight_bit

- accelerate launch --num_processes 8 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/cot_gsm8k --logging_path logs/gpt2_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 128 --model_path gpt2 --tok_path gpt2

- accelerate launch --num_processes 8 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/cot_gsm8k --logging_path logs/vicuna_7B_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 256 --model_path /private/home/alexdahoas/repos/trlx/examples/experiments/in_context/ckpts/vicuna_7B_gsm --K 8

- accelerate launch --num_processes 8 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/cot_gsm8k --logging_path logs/vicuna_7B_test_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 256 --model_path /private/home/alexdahoas/repos/trlx/examples/experiments/in_context/ckpts/vicuna_7B_gsm --K 8 --split test

- accelerate launch --num_processes 8 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/gsm_socratic_conditional --logging_path logs/socratic_conditional_vicuna_7B_test_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 256 --model_path /private/home/alexdahoas/repos/trlx/examples/experiments/in_context/ckpts/vicuna_7B_gsm_socratic_conditional --K 8 --split test

- accelerate launch --num_processes 8 --config_file configs/infer.yaml infer.py --prompt_dataset Dahoas/cot_gsm8k_socratic --logging_path logs/socratic_vicuna_7B_test_answers.jsonl --batch_size 1 --temp 0.7 --max_new_tokens 256 --model_path /private/home/alexdahoas/repos/trlx/examples/experiments/in_context/ckpts/socratic_vicuna_7B_gsm_z3 --K 8 --split test

## Eval

- python eval.py --eval_dataset logs/merged_vicuna_7B_answers.jsonl \
--logging_path logs/merged_vicuna_7B_test_answers_heuristic_scores.jsonl --mode sparse --bon_scheme voting

- python eval.py --eval_dataset logs/merged_socratic_vicuna_7B_test_answers.jsonl \
 --mode heuristic --bon_scheme max

## Debugging

 - accelerate launch --num_processes 2 --config_file ../configs/accelerate/zero3.yaml ppo_sentiments.py
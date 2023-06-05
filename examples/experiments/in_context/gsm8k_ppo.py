import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from eval import eval_answer

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=600,
        epochs=10000,
        total_steps=10000,
        batch_size=4,
        checkpoint_interval=100000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
    ),
    model=ModelConfig(
         model_path="/private/home/alexdahoas/repos/trlx/examples/experiments/in_context/ckpts/vicuna_7B_gsm", 
         num_layers_unfrozen=16,
         delta_kwargs=dict(
              delta_type="lora",
              modified_modules="all",
              lora_r=32,
              lora_alpha=16,
              lora_dropout=0.0,
         ),
    ),
    tokenizer=TokenizerConfig(tokenizer_path="oobabooga/llama-tokenizer", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        ),
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    save_path = "ckpts/vicuna_7B_gsm_ppo_1"
    #config.model.delta_kwargs = None
    #config.model.model_path = "gpt2"
    #config.tokenizer.tokenizer_path = "gpt2"

    dataset = load_dataset("Dahoas/cot_gsm8k")
    prompts = [{"prompt": x["prompt"], "gt_answer": x["answer"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"], "gt_answer": x["answer"]} for x in dataset["val"]][:8]
    def reward_fn(outputs, gt_answer, *args, **kwargs):
        return [float(eval_answer(output, gt_ans)[0]) for output, gt_ans in zip(outputs, gt_answer)]

    trainer = trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Q:", "A:"],
    )
    trainer.save_pretrained(save_path)


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

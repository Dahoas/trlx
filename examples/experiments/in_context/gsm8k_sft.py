import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

'''delta_kwargs=dict(
              delta_type="lora",
              modified_modules="all",
              lora_r=16,
              lora_alpha=16,
              lora_dropout=0.0,
         ),'''

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=800,
        epochs=1,
        total_steps=50000,
        checkpoint_interval=10000000,
        batch_size=1,
        eval_interval=50000,
        pipeline="PromptPipeline",
        trainer="AccelerateSFTTrainer",
    ),
    model=ModelConfig(
        model_path="/private/home/alexdahoas/repos/ckpts/vicuna_7B",#"facebook/galactica-6.7b",#"llama_7B_gsm",#/private/home/alexdahoas/repos/ckpts/llama_7B/", 
        num_layers_unfrozen=-1,
    ),
    tokenizer=TokenizerConfig(tokenizer_path="oobabooga/llama-tokenizer", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=100000000, eta_min=1e-6)),
    method=SFTConfig(
        name="sftconfig",
        gen_kwargs=dict(max_new_tokens=625, do_sample=False), #top_p=1.0, temperature=0.7, do_sample=True),
    ),
)


def preprocess(sample):
    sample["full"] = sample["prompt"] + sample["response"]
    return sample
    

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    dataset_path = "Dahoas/gsm_socratic_conditional"#Dahoas/cot_gsm8k"
    dataset = load_dataset(dataset_path).map(preprocess)#load_jsonl("logs/socratic_conditional_finetune.jsonl")#load_dataset("Dahoas/cot_gsm8k").map(preprocess)
    save_path = "vicuna_7B_gsm_socratic_conditional/"

    tok = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)

    def metric_fn(samples, **kwargs):
        scores = [float(sample.split(tok.eos_token)[0].split("#### ")[-1] == gt.split(tok.eos_token)[0].split("#### ")[1]) for sample, gt in zip(samples, kwargs["response"])]
        return {"scores": scores}

    trainer = trlx.train(
        config=config,
        samples=dataset["train"]["full"],
        eval_prompts=dataset.filter(lambda s, i: i < 8, with_indices=True)["val"],
        metric_fn=metric_fn,
        stop_sequences=["Q: ", "A: "],
    )
    trainer.save_pretrained(save_path)


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

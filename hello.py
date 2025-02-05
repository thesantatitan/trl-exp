# First, let's import our required libraries
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Let's use GPT-2 small for this example
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cpu")
model = model.to(device)


import re
import vllm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl.trl.trainer.grpo_trainer import GRPOTrainer
from trl.trl.trainer.grpo_config import GRPOConfig

import pandas as pd
dataset = pd.read_parquet("https://huggingface.co/datasets/PleIAs/verse-wikisource/resolve/main/verse_wikisource.parquet")

prompt_list = []

for verse in dataset["verse"].tolist()[0:1000]:
  prompt_list.append(f"{verse}\n")

dataset = Dataset.from_dict({'prompt': prompt_list})

dataset

output_dir="outputs/Pleias-350m-GRPO"
run_name="Pleias-350m-GRPO-Poetry"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cpu",
    report_to="none" #I'm disabling Wandb.
)

def no_repetition_reward_func(completions, hey) -> list[float]:
    # Handle both string and conversational formats
    responses = []
    for completion in completions:
        if isinstance(completion, str):
            responses.append(completion)
        elif isinstance(completion, list):
            responses.append(completion[0]["content"])
        else:
            raise ValueError(f"Unexpected completion format: {type(completion)}")

    # Calculate continuous scores
    scores = [1.0 for response in responses]

    return scores

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        no_repetition_reward_func,
    ],
    args=training_args,
    train_dataset=dataset
)
trainer.train()

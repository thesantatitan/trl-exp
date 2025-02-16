import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import cairosvg
import io
from PIL import Image
from typing import List, Tuple, Optional, Dict
import torch
import clip
from rewards import SVGRewardFunction
from wandbtracker import WandbPredictionProgressCallback

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
...
<generated_svg>
...
</generated_svg>
"""

def prep_dataset(num_rows: int = None) -> Dataset:
    # Initialize SVGRewardFunction
    reward_function = SVGRewardFunction()

    # Load dataset with optional row limit
    dataset = load_dataset(
        "thesantatitan/svg-rendered-blip_captioned",
        split="train"
    )

    # If num_rows is specified, select only that many rows
    if num_rows is not None:
        dataset = dataset.select(range(min(num_rows, len(dataset))))

    def process_example(example):
        # Create prompt
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"generate svg code for an image that looks like {example['caption']}"},
            {'role': 'assistant', 'content': "<reasoning>"}
        ]
        if not example['png_processed']:
            return
        try:
            # Process image using SVGRewardFunction's methods
            image = Image.open(io.BytesIO(example['png_data'])).convert('RGB')
            image_input = reward_function.preprocess(image).unsqueeze(0).to(reward_function.device)

            with torch.no_grad():
                # Get image embeddings using the reward function's CLIP model
                image_features = reward_function.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Get text embeddings using the reward function's methods
                text_features = reward_function.encode_text([example['caption']])

                return {
                    'prompt': prompt,
                    'ground_truth_embeddings': image_features.cpu(),
                    'text_embeddings': text_features.cpu()
                }
        except Exception as e:
            print(f"Error processing example: {e}")
            return {
                'prompt': prompt,
                'ground_truth_embeddings': torch.zeros(1, 512),
                'text_embeddings': torch.zeros(1, 512)
            }

    # Process the dataset
    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset.features.keys(),
        desc="Processing dataset",
        num_proc=1  # Set to higher number for parallel processing if needed
    )
    del reward_function

    return processed_dataset

dataset_svg = prep_dataset(1000)

svg_reward_fn = SVGRewardFunction(
    format_weight=1.0,
    rendering_weight=2.0,
    clip_weight=4.0,
    text_weight=4.0,
    device="cuda"
)

wandb_callback = WandbPredictionProgressCallback(
    reward_func=svg_reward_fn
)

model_name = "thesantatitan/Qwen2-0.5B-svg-SFT"

output_dir="outputs/Qwen-0.5B-GRPO"
run_name="Qwen-0.5B-GRPO-svg-after-sft"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=512,
    max_completion_length=4096,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=40,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
    vllm_dtype="half",
    vllm_device="cuda:0",
    report_to="none",
    hub_strategy="every_save",
    push_to_hub=True,
    eval_strategy="no",
    torch_empty_cache_steps=10
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None,
    attn_implementation="flash_attention_2"
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=svg_reward_fn,
    args=training_args,
    train_dataset=dataset_svg,
    callbacks=[wandb_callback],
    #peft_config=peft_config
)
trainer.train()

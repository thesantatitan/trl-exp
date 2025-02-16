import os
import json
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm
import concurrent.futures

# --------- CONFIGURATION ---------
NUM_SAMPLES = 5000
DATASET_NAME = "thesantatitan/svg-rendered-blip_captioned"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # Updated API endpoint
MODEL_NAME = "deepseek/deepseek-r1-distill-qwen-32b"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set your OpenRouter API key as the environment variable OPENROUTER_API_KEY.")

# Destination repo on the Hugging Face Hub (format: "username/repo_name")
DEST_HF_REPO = "thesantatitan/deepseek-svg-dataset"  # Update with your repository id.

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENROUTER_API_KEY}"
}

# --------- FUNCTIONS ---------
def build_prompt(caption: str) -> str:
    """
    Build a prompt asking the model to generate SVG code for an image described by the caption.
    """
    prompt = f"Generate svg code for an image that looks like: {caption}. Don't use markdown just give svg code\n"
    return prompt

def call_deepseek(prompt: str) -> str:
    """
    Send a POST request to the OpenRouter API with the given prompt.
    Returns the model's response (the SVG code and reasoning) as a string.
    """
    payload = {
        "model": MODEL_NAME,
        # Use only the user message as in the provided sample code.
        "messages": [{"role":"system", "content":"Respond in the format <generated_svg>{svg code}</generated_svg>"},{"role": "user", "content": prompt}],
        "include_reasoning": True
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        # Uncomment the following line if you want to view the full response for debugging.
        # print(json.dumps(data, indent=2))
        if "choices" in data and len(data["choices"]) > 0:
            # Return both reasoning and SVG code.
            return f'''
<reasoning>
{data["choices"][0]["message"]["reasoning"]}
</reasoning>
{data["choices"][0]["message"]["content"]}
            '''
        else:
            return "No valid response from model."
    except Exception as e:
        print(f"Request failed: {e}")
        return f"Error: {e}"

def process_sample(sample):
    """
    Process a single sample: build the prompt and call the API.
    Returns a tuple (prompt, completion).
    """
    caption = sample.get("caption", "")
    prompt = build_prompt(caption)
    completion = call_deepseek(prompt)
    return prompt, completion

# --------- MAIN SCRIPT ---------
def main():
    print("Loading dataset in streaming mode...")
    ds_stream = load_dataset(DATASET_NAME, split="train", streaming=True)

    print(f"Skipping some samples and taking {NUM_SAMPLES} random samples...")
    # Convert the streaming iterator to a list. We skip the first 10000 samples and then take NUM_SAMPLES.
    samples = list(ds_stream.skip(10000).take(NUM_SAMPLES))
    print("Samples obtained.")

    prompts = []
    completions = []

    # Use a ThreadPoolExecutor to make parallel API calls.
    max_workers = 128  # You can adjust this number based on your environment
    print("Processing each caption and querying the model in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Using list() to force evaluation, and tqdm for progress tracking.
        results = list(tqdm(executor.map(process_sample, samples), total=len(samples)))

    # Unpack the results
    for prompt, response_text in results:
        prompts.append(prompt)
        completions.append(response_text)

    print("Building a new dataset with 'prompt' and 'completion' columns...")
    new_dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})

    # Save the dataset locally (optional) then push it to the Hugging Face Hub.
    tmp_path = "deepseek_svg_dataset.arrow"
    new_dataset.save_to_disk(tmp_path)
    print(f"Dataset saved locally to: {tmp_path}")

    print(f"Pushing the new dataset to the Hugging Face Hub at repo '{DEST_HF_REPO}'...")
    new_dataset.push_to_hub(DEST_HF_REPO, private=False)  # Set private=True if desired.

    print("Done!")

if __name__ == "__main__":
    main()

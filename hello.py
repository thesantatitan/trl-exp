from datasets import load_dataset
from PIL import Image
import io
import torch
import clip
from typing import Dict

def prep_dataset(num_examples: int = 1000) -> Dict:
    # System prompt for SVG generation
    SYSTEM_PROMPT = """You are an expert at generating SVG code.
    Your task is to generate SVG code that matches the given description.
    Wrap your response in the following format:
    <think>
    [Your thoughts about how to approach this]
    </think>
    [Any additional context or explanation]
    <svg>[Your SVG code here]</svg>"""

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load dataset with streaming
    dataset = load_dataset(
        "thesantatitan/svg-rendered-blip_captioned",
        split="train",
        streaming=True
    )

    def process_example(example):
        # Create prompt
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"generate svg code for an image that looks like {example['caption']}"}
        ]

        # Process image for CLIP
        try:
            image = Image.open(io.BytesIO(example['png_data'])).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                # Get image embeddings
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Get text embeddings
                text_tokens = clip.tokenize([example['caption']]).to(device)
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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

    # Process the streaming dataset
    processed_dataset = dataset.take(num_examples).map(
        process_example,
        remove_columns=dataset.features.keys()
    )

    return processed_dataset

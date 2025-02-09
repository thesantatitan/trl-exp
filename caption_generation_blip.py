import io
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import torch
import pandas as pd
import os

def setup_model():
    """Initialize the BLIP model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('mps')
    return processor, model

def generate_captions_batch(images, processor, model, batch_size=32):
    """Generate captions for a batch of images."""
    captions = []

    # Process images in smaller batches
    for i in tqdm(range(0, len(images), batch_size), desc="Generating captions", leave=False):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        else:
            inputs = {k: v.to('mps') for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs)

        # Decode captions
        batch_captions = [processor.decode(o, skip_special_tokens=True) for o in out]
        captions.extend(batch_captions)

    return captions

def process_dataset_chunk(dataset_chunk, processor, model, caption_batch_size):
    """Process a chunk of the dataset and generate captions."""
    images = []
    valid_indices = []

    # Get lists from the dataset chunk
    png_data_list = dataset_chunk['png_data']
    png_processed_list = dataset_chunk['png_processed']

    # Convert binary data to PIL Images
    for idx in tqdm(range(len(png_data_list)), desc="Processing images", leave=False):
        if png_processed_list[idx] and png_data_list[idx]:
            try:
                img = Image.open(io.BytesIO(png_data_list[idx])).convert('RGB')
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                continue

    # Generate captions for valid images
    if images:
        captions = generate_captions_batch(images, processor, model, caption_batch_size)

        # Create a new dataset with captions
        # Initialize a dictionary with lists for all fields
        processed_data = {key: list(values) for key, values in dataset_chunk.items()}

        # Add new caption field
        processed_data['caption'] = [None] * len(png_data_list)

        # Fill in captions for valid indices
        for valid_idx, caption in zip(valid_indices, captions):
            processed_data['caption'][valid_idx] = caption

        return processed_data

    # If no valid images, return original data with None captions
    processed_data = {key: list(values) for key, values in dataset_chunk.items()}
    processed_data['caption'] = [None] * len(png_data_list)
    return processed_data

def main(chunk_size=1000, caption_batch_size=32, output_dir="processed_datasets"):
    """Main function to process the dataset."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("thesantatitan/svg-rendered", split="train")
    total_chunks = (len(dataset) + chunk_size - 1) // chunk_size

    # Setup model
    print("Setting up model...")
    processor, model = setup_model()

    # Process dataset in chunks
    for i in tqdm(range(0, len(dataset), chunk_size), desc="Processing chunks", total=total_chunks):
        chunk = dataset[i:i + chunk_size]

        # Process chunk and generate captions
        processed_chunk = process_dataset_chunk(chunk, processor, model, caption_batch_size)

        # Convert processed chunk to Dataset object
        processed_dataset = Dataset.from_dict(processed_chunk)

        # Save processed chunk as a dataset
        processed_dataset.save_to_disk(os.path.join(output_dir, f"chunk_{i//chunk_size}"))

if __name__ == "__main__":
    # You can adjust these parameters
    CHUNK_SIZE = 1024  # Size of dataset chunks
    CAPTION_BATCH_SIZE = 8 # Size of batches for caption generation
    OUTPUT_DIR = "processed_datasets"

    main(CHUNK_SIZE, CAPTION_BATCH_SIZE, OUTPUT_DIR)

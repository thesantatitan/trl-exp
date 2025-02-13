import os
import time
import argparse
from functools import wraps
from datasets import load_dataset, Dataset
import logging
import io
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from tqdm.auto import tqdm
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        if args and hasattr(args[0], '__class__'):
            if not hasattr(args[0], 'function_times'):
                args[0].function_times = {}
            if func.__name__ not in args[0].function_times:
                args[0].function_times[func.__name__] = []
            args[0].function_times[func.__name__].append(duration)
        logger.info(f"{func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

class SmolVLMCaptioner:
    def __init__(self, source_repo: str, batch_index: int, total_batches: int, token: str):
        self.source_repo = source_repo
        self.batch_index = batch_index
        self.total_batches = total_batches
        self.token = token
        self.function_times = {}
        self.processed_count = 0

        # Initialize SmolVLM model
        logger.info("Initializing SmolVLM model...")
        start_time = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        ).to(self.device)
        logger.info(f"SmolVLM initialization took {time.time() - start_time:.2f} seconds")

    @timing_decorator
    def load_dataset(self) -> Dataset:
        try:
            dataset_info = load_dataset(self.source_repo, split='train', streaming=True)
            total_rows = dataset_info.info.splits['train'].num_examples

            rows_per_batch = math.ceil(total_rows / self.total_batches)
            start_idx = self.batch_index * rows_per_batch
            end_idx = min(start_idx + rows_per_batch, total_rows)

            logger.info(f"Processing batch {self.batch_index + 1}/{self.total_batches}")
            logger.info(f"Rows {start_idx} to {end_idx} (total: {end_idx - start_idx})")

            dataset = load_dataset(
                self.source_repo,
                split='train',
                streaming=True
            )

            batch_dataset = dataset.skip(start_idx).take(end_idx - start_idx)
            return Dataset.from_generator(lambda: batch_dataset)

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @timing_decorator
    def generate_caption(self, png_data) -> dict:
        """Generate caption for a single image using SmolVLM."""
        try:
            if png_data is None:
                return {'smolvlm_caption': "Failed - No valid PNG data"}

            image = Image.open(io.BytesIO(png_data))

            # Create input messages for SmolVLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                }
            ]

            # Prepare inputs
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            return {'smolvlm_caption': generated_text}

        except Exception as e:
            logger.error(f"Error generating caption for image: {e}")
            return {'smolvlm_caption': "Error processing image"}

    def process_row(self, row):
        """Process a single row from the dataset."""
        try:
            self.processed_count += 1
            start_time = time.time()

            if not row['png_processed'] or row['png_data'] is None:
                result = {'smolvlm_caption': "Failed - No valid PNG data"}
            else:
                result = self.generate_caption(row['png_data'])

            processing_time = time.time() - start_time

            if self.processed_count % 50 == 0:
                avg_times = {
                    func: sum(times)/len(times)
                    for func, times in self.function_times.items()
                }
                logger.info(
                    f"Processed {self.processed_count} items\n"
                    f"Average times:\n"
                    f"  generate_caption: {avg_times.get('generate_caption', 0):.2f}s\n"
                    f"  Total per row: {processing_time:.2f}s"
                )

            return result

        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return {'smolvlm_caption': "Error processing row"}

    @timing_decorator
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Add caption columns to the dataset."""
        try:
            logger.info("Adding caption columns")
            total_rows = len(dataset)

            pbar = tqdm(total=total_rows, desc=f"Generating captions (Batch {self.batch_index + 1}/{self.total_batches})")

            def update_progress(_, processed_rows):
                pbar.update(processed_rows)

            processed_dataset = dataset.map(
                self.process_row,
                remove_columns=dataset.column_names
            )

            pbar.close()
            return processed_dataset

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

    @timing_decorator
    def save_dataset(self, dataset: Dataset) -> None:
        """Save the processed dataset locally."""
        try:
            output_dir = f"/dataset/{self.batch_index}"
            os.makedirs(output_dir, exist_ok=True)
            dataset.save_to_disk(output_dir)
            logger.info(f"Successfully saved batch to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process dataset in batches')
    parser.add_argument('--number_of_batches', type=int, required=True,
                      help='Total number of parallel processes')
    parser.add_argument('--index', type=int, required=True,
                      help='Index of current batch (0-based)')
    args = parser.parse_args()

    if args.index >= args.number_of_batches:
        raise ValueError("Batch index must be less than number of batches")

    SOURCE_REPO = "thesantatitan/svg-rendered"
    TOKEN = os.getenv("HF_TOKEN")

    if TOKEN is None:
        raise ValueError("HF_TOKEN environment variable not set")

    start_time = time.time()
    captioner = SmolVLMCaptioner(
        SOURCE_REPO,
        args.index,
        args.number_of_batches,
        TOKEN
    )
    logger.info(f"Initialization took {time.time() - start_time:.2f} seconds")

    try:
        dataset = captioner.load_dataset()
        processed_dataset = captioner.process_dataset(dataset)
        captioner.save_dataset(processed_dataset)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

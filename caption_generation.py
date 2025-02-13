import os
import time
import argparse
from functools import wraps
from datasets import load_dataset, Dataset
import logging
import io
from PIL import Image
from transformers import AutoModelForCausalLM
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

class MoondreamCaptioner:
    def __init__(self, source_repo: str, batch_size: int, token: str):
        self.source_repo = source_repo
        self.batch_size = batch_size
        self.token = token
        self.function_times = {}
        self.processed_count = 0
        self.BATCH_SIZE = 8  # Processing batch size for GPU operations

        # Initialize Moondream model
        logger.info("Initializing Moondream model...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": "mps"}
        )
        logger.info(f"Moondream initialization took {time.time() - start_time:.2f} seconds")

    @timing_decorator
    def load_dataset(self) -> Dataset:
        try:
            logger.info(f"Loading dataset from {self.source_repo}")
            dataset = load_dataset(self.source_repo, split='train')
            total_rows = len(dataset)
            logger.info(f"Total dataset size: {total_rows} rows")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @timing_decorator
    def generate_captions_batch(self, png_data_list: list) -> list:
        """Generate both short and normal captions for a batch of images."""
        results = []
        for png_data in png_data_list:
            try:
                if png_data is None:
                    results.append(({
                        'moondream_short_caption': "Failed - No valid PNG data",
                        'moondream_normal_caption': "Failed - No valid PNG data"
                    }))
                    continue

                image = Image.open(io.BytesIO(png_data))
                short_caption = self.model.caption(image, length="short")["caption"]
                normal_caption = self.model.caption(image)["caption"]
                results.append({
                    'moondream_short_caption': short_caption,
                    'moondream_normal_caption': normal_caption
                })
            except Exception as e:
                logger.error(f"Error generating captions for image in batch: {e}")
                results.append({
                    'moondream_short_caption': "Error processing image",
                    'moondream_normal_caption': "Error processing image"
                })
        return results

    def process_batch(self, batch):
        """Process a batch of rows from the dataset."""
        try:
            self.processed_count += len(batch['png_data'])
            start_time = time.time()

            # Filter valid PNG data
            valid_png_data = [
                data for data, is_processed in zip(batch['png_data'], batch['png_processed'])
                if is_processed and data is not None
            ]

            # Generate captions for valid images
            caption_results = self.generate_captions_batch(valid_png_data)

            # Prepare results for all images (including invalid ones)
            all_results = []
            result_idx = 0
            for is_processed, data in zip(batch['png_processed'], batch['png_data']):
                if is_processed and data is not None:
                    all_results.append(caption_results[result_idx])
                    result_idx += 1
                else:
                    all_results.append({
                        'moondream_short_caption': "Failed - No valid PNG data",
                        'moondream_normal_caption': "Failed - No valid PNG data"
                    })

            processing_time = time.time() - start_time

            # Log timing information every 50 items
            if self.processed_count % 50 == 0:
                avg_times = {
                    func: sum(times)/len(times)
                    for func, times in self.function_times.items()
                }
                logger.info(
                    f"Processed {self.processed_count} items\n"
                    f"Average times:\n"
                    f"  generate_captions_batch: {avg_times.get('generate_captions_batch', 0):.2f}s\n"
                    f"  Total per batch: {processing_time:.2f}s"
                )

            return {
                'moondream_short_caption': [r['moondream_short_caption'] for r in all_results],
                'moondream_normal_caption': [r['moondream_normal_caption'] for r in all_results]
            }

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {
                'moondream_short_caption': ["Error processing batch"] * len(batch['png_data']),
                'moondream_normal_caption': ["Error processing batch"] * len(batch['png_data'])
            }

    @timing_decorator
    def process_dataset(self, dataset: Dataset) -> list:
        """Process the dataset in batches and save each batch."""
        try:
            total_rows = len(dataset)
            batches = math.ceil(total_rows / self.batch_size)
            logger.info(f"Processing dataset in {batches} batches")

            processed_datasets = []

            for batch_idx in range(batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_rows)

                logger.info(f"Processing batch {batch_idx + 1}/{batches}")
                logger.info(f"Rows {start_idx} to {end_idx}")

                # Get current batch
                current_batch = dataset.select(range(start_idx, end_idx))

                # Process the batch
                processed_batch = current_batch.map(
                    self.process_batch,
                    batched=True,
                    batch_size=self.BATCH_SIZE,
                    num_proc=1,  # Single process due to GPU usage
                    remove_columns=current_batch.column_names
                )

                # Save the batch
                self.save_batch(processed_batch, batch_idx)
                processed_datasets.append(processed_batch)

            # return processed_datasets

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

    @timing_decorator
    def save_batch(self, dataset: Dataset, batch_idx: int) -> None:
        """Save a processed batch locally."""
        try:
            output_dir = f"./dataset/batch_{batch_idx}"
            os.makedirs(output_dir, exist_ok=True)
            dataset.save_to_disk(output_dir)
            logger.info(f"Successfully saved batch to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving batch: {e}")
            raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process dataset in batches')
    parser.add_argument('--number_of_batches', type=int, required=True,
                      help='Number of batches to split the dataset into')
    args = parser.parse_args()

    SOURCE_REPO = "thesantatitan/svg-rendered"
    TOKEN = os.getenv("HF_TOKEN")

    start_time = time.time()
    captioner = MoondreamCaptioner(
        SOURCE_REPO,
        args.number_of_batches,
        TOKEN
    )
    logger.info(f"Initialization took {time.time() - start_time:.2f} seconds")

    try:
        # Load dataset
        dataset = captioner.load_dataset()

        # Process dataset in batches
        captioner.process_dataset(dataset)

        logger.info("Successfully processed all batches")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

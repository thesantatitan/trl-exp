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
    def __init__(self, source_repo: str, batch_index: int, total_batches: int, token: str):
        self.source_repo = source_repo
        self.batch_index = batch_index
        self.total_batches = total_batches
        self.token = token
        self.function_times = {}
        self.processed_count = 0
        self.BATCH_SIZE = 8  # New constant for processing batch size

        # Initialize Moondream model
        logger.info("Initializing Moondream model...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": "cuda"}
        )
        logger.info(f"Moondream initialization took {time.time() - start_time:.2f} seconds")

    @timing_decorator
    def load_dataset(self) -> Dataset:
        try:
            # First, get the total dataset size without downloading
            logger.info(f"Getting dataset info from {self.source_repo}")
            dataset_info = load_dataset(self.source_repo, split='train', streaming=True)
            total_rows = dataset_info.info.splits['train'].num_examples

            # Calculate batch boundaries
            rows_per_batch = math.ceil(total_rows / self.total_batches)
            start_idx = self.batch_index * rows_per_batch
            end_idx = min(start_idx + rows_per_batch, total_rows)

            logger.info(f"Processing batch {self.batch_index + 1}/{self.total_batches}")
            logger.info(f"Rows {start_idx} to {end_idx} (total: {end_idx - start_idx})")

            # Load only the required slice using streaming and take
            dataset = load_dataset(
                self.source_repo,
                split='train',
                streaming=True
            )

            # Skip to our section and take only what we need
            batch_dataset = dataset.skip(start_idx).take(end_idx - start_idx)

            # Convert streaming dataset to regular dataset for processing
            return Dataset.from_generator(lambda: batch_dataset)

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

            # Combine results into a single dictionary for the batch
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
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Add caption columns to the dataset using batched processing."""
        try:
            logger.info("Adding caption columns with batch processing")
            total_rows = len(dataset)

            pbar = tqdm(total=total_rows, desc=f"Generating captions (Batch {self.batch_index + 1}/{self.total_batches})")

            def update_progress(_, processed_rows):
                pbar.update(processed_rows)

            processed_dataset = dataset.map(
                self.process_batch,
                batched=True,
                batch_size=self.BATCH_SIZE,
                num_proc=1,  # Single process due to GPU usage
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
            # Create output directory if it doesn't exist
            output_dir = f"/dataset/{self.batch_index}"
            os.makedirs(output_dir, exist_ok=True)

            # Save the dataset to the specified directory
            dataset.save_to_disk(output_dir)
            logger.info(f"Successfully saved batch to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

def main():
    # Parse command line arguments
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
    captioner = MoondreamCaptioner(
        SOURCE_REPO,
        args.index,
        args.number_of_batches,
        TOKEN
    )
    logger.info(f"Initialization took {time.time() - start_time:.2f} seconds")

    try:
        # Load dataset
        dataset = captioner.load_dataset()

        # Process dataset
        processed_dataset = captioner.process_dataset(dataset)

        # Save locally
        captioner.save_dataset(processed_dataset)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

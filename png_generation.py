import os
import time
from functools import wraps
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, login
from typing import List, Dict, Any
import logging
import cairosvg
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import pathlib

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

class DatasetProcessor:
    def __init__(self, source_repo: str, target_repo: str, token: str, output_dir: str, batch_size: int = 32):
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.token = token
        self.output_dir = pathlib.Path(output_dir)
        self.api = HfApi()
        self.function_times = {}
        self.processed_count = 0
        self.batch_size = batch_size

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Login to Hugging Face
        login(token=token)

    def _convert_single_svg(self, svg_string: str) -> tuple[bool, bytes | None]:
        """Convert a single SVG to PNG with proper error handling."""
        try:
            # Validate input
            if not svg_string or not isinstance(svg_string, str):
                logger.warning(f"Invalid SVG input: {type(svg_string)}")
                return False, None

            # Convert SVG to PNG bytes
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))

            # Validate the PNG data by attempting to open it
            try:
                img = Image.open(io.BytesIO(png_data))
                img.verify()  # Verify it's a valid image
                return True, png_data
            except Exception as e:
                logger.error(f"Invalid PNG data generated: {e}")
                return False, None

        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {e}")
            return False, None

    @timing_decorator
    def batch_svg_to_png(self, svg_strings: List[str]) -> List[tuple[bool, bytes | None]]:
        """Convert a batch of SVGs to PNGs maintaining batch size."""
        with ThreadPoolExecutor(max_workers=min(len(svg_strings), 8)) as executor:
            results = list(executor.map(self._convert_single_svg, svg_strings))
        return results

    def process_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of examples maintaining batch size."""
        start_time = time.time()

        # Convert SVGs to PNGs in batch
        conversion_results = self.batch_svg_to_png(batch['output'])

        # Separate success flags and PNG data while maintaining batch size
        success_flags = []
        png_data = []

        for success, data in conversion_results:
            success_flags.append(success)
            png_data.append(data if success else None)

        # Update progress metrics
        self.processed_count += len(batch['output'])
        if self.processed_count % (self.batch_size * 2) == 0:
            conversion_rate = sum(success_flags) / len(success_flags) * 100
            avg_times = {
                func: sum(times)/len(times)
                for func, times in self.function_times.items()
            }
            logger.info(
                f"Processed {self.processed_count} items\n"
                f"Conversion success rate: {conversion_rate:.1f}%\n"
                f"Average batch processing time: {avg_times.get('batch_svg_to_png', 0):.2f}s"
            )

        # Return processed batch with both success flags and PNG data
        return {
            **batch,
            'png_processed': success_flags,
            'png_data': png_data
        }

    @timing_decorator
    def load_dataset(self, split: str = "train") -> Dataset:
        try:
            logger.info(f"Loading dataset from {self.source_repo}")
            dataset = load_dataset(self.source_repo, split=split, revision="refs/convert/parquet")
            logger.info(f"Successfully loaded dataset with {len(dataset)} rows")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @timing_decorator
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process the entire dataset with batching."""
        try:
            logger.info("Processing dataset in batches")
            return dataset.map(
                self.process_batch,
                batched=True,
                batch_size=self.batch_size,
                remove_columns=dataset.column_names,
                desc="Processing SVGs in batches"
            )
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

    @timing_decorator
    def save_dataset_to_disk(self, dataset: Dataset) -> None:
        """Save the processed dataset to disk."""
        try:
            logger.info(f"Saving dataset to disk at {self.output_dir}")
            dataset.save_to_disk(self.output_dir)
            logger.info("Successfully saved dataset to disk")
        except Exception as e:
            logger.error(f"Error saving dataset to disk: {e}")
            raise

    @timing_decorator
    def create_and_push_dataset(self, dataset: Dataset) -> None:
        try:
            try:
                create_repo(self.target_repo, private=False, token=self.token)
                logger.info(f"Created new repository: {self.target_repo}")
            except Exception as e:
                logger.warning(f"Repository creation warning (might already exist): {e}")

            logger.info("Pushing dataset to hub...")
            dataset.push_to_hub(
                self.target_repo,
                private=False,
                token=self.token
            )
            logger.info(f"Successfully pushed dataset to {self.target_repo}")
        except Exception as e:
            logger.error(f"Error pushing dataset: {e}")
            raise

def main():
    SOURCE_REPO = "umuthopeyildirim/svgen-500k-instruct"
    TARGET_REPO = "thesantatitan/svg-rendered"
    OUTPUT_DIR = "processed_dataset"  # Local directory to save the dataset
    TOKEN = os.getenv("HF_TOKEN")
    BATCH_SIZE = 56  # Configurable batch size

    processor = DatasetProcessor(SOURCE_REPO, TARGET_REPO, TOKEN, OUTPUT_DIR, BATCH_SIZE)

    try:
        dataset = processor.load_dataset()
        processed_dataset = processor.process_dataset(dataset)

        # Save to disk first
        processor.save_dataset_to_disk(processed_dataset)

        # Then upload to HuggingFace
        processor.create_and_push_dataset(processed_dataset)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

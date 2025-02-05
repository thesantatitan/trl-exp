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
import moondream as md
from tqdm.auto import tqdm

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        # Get the instance (self) if it's a method
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
    def __init__(self, source_repo: str, target_repo: str, token: str, moondream_path: str):
        """
        Initialize the dataset processor.
        """
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.token = token
        self.api = HfApi()
        self.function_times = {}  # Store timing information
        self.processed_count = 0

        # Initialize Moondream model
        logger.info("Initializing Moondream model...")
        start_time = time.time()
        self.model = md.vl(model=moondream_path)
        logger.info(f"Moondream initialization took {time.time() - start_time:.2f} seconds")

        # Login to Hugging Face
        login(token=token)

    @timing_decorator
    def svg_to_png(self, svg_string: str) -> Image.Image:
        """Convert SVG string to PNG image."""
        try:
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
            image = Image.open(io.BytesIO(png_data))
            return image
        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {e}")
            raise

    @timing_decorator
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image using Moondream."""
        try:
            encoded_image = self.model.encode_image(image)
            caption = self.model.caption(encoded_image)["caption"]
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            raise

    def custom_processing(self, row: Dict[str, Any]) -> Any:
        """Process SVG input with timing information."""
        try:
            self.processed_count += 1
            start_time = time.time()

            # Time SVG to PNG conversion
            svg_string = row['output']
            image = self.svg_to_png(svg_string)

            # Time caption generation
            caption = self.generate_caption(image)

            total_time = time.time() - start_time

            # Log timing information every 10 items
            if self.processed_count % 10 == 0:
                avg_times = {
                    func: sum(times)/len(times)
                    for func, times in self.function_times.items()
                }
                logger.info(
                    f"Processed {self.processed_count} items\n"
                    f"Average times:\n"
                    f"  svg_to_png: {avg_times.get('svg_to_png', 0):.2f}s\n"
                    f"  generate_caption: {avg_times.get('generate_caption', 0):.2f}s\n"
                    f"  Total per item: {total_time:.2f}s"
                )

            return caption
        except Exception as e:
            logger.error(f"Error in custom processing: {e}")
            return "Error processing image"

    @timing_decorator
    def load_dataset(self, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace."""
        try:
            logger.info(f"Loading dataset from {self.source_repo}")
            dataset = load_dataset(self.source_repo, split=split, revision="refs/convert/parquet")
            logger.info(f"Successfully loaded dataset with {len(dataset)} rows")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @timing_decorator
    def extract_columns(self, dataset: Dataset, columns: List[str]) -> Dataset:
        """Extract specific columns from dataset."""
        try:
            logger.info(f"Extracting columns: {columns}")
            return dataset.select_columns(columns)
        except Exception as e:
            logger.error(f"Error extracting columns: {e}")
            raise

    @timing_decorator
    def add_custom_column(self, dataset: Dataset,
                         new_column_name: str,
                         processing_func: callable) -> Dataset:
        """Add a new column based on existing data."""
        try:
            logger.info(f"Adding new column: {new_column_name}")
            total_rows = len(dataset)

            # Initialize progress bar with additional metrics
            pbar = tqdm(total=total_rows, desc="Processing images")

            def map_function(example):
                start_time = time.time()
                example[new_column_name] = processing_func(example)
                processing_time = time.time() - start_time

                # Update progress bar with timing information
                pbar.set_postfix({
                    'svg_to_png_avg': f"{sum(self.function_times.get('svg_to_png', [0]))/len(self.function_times.get('svg_to_png', [1])):.2f}s",
                    'caption_avg': f"{sum(self.function_times.get('generate_caption', [0]))/len(self.function_times.get('generate_caption', [1])):.2f}s",
                    'total_avg': f"{processing_time:.2f}s"
                })
                pbar.update(1)
                return example

            processed_dataset = dataset.map(
                map_function,
                batched=False,
                num_proc=1
            )

            pbar.close()
            return processed_dataset

        except Exception as e:
            logger.error(f"Error adding custom column: {e}")
            raise

    @timing_decorator
    def create_and_push_dataset(self, dataset: Dataset) -> None:
        """Create new repo and push processed dataset."""
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
    # Configuration
    SOURCE_REPO = "umuthopeyildirim/svgen-500k-instruct"
    TARGET_REPO = "thesantatitan/tempm"
    TOKEN = os.getenv("HF_TOKEN")
    COLUMNS_TO_EXTRACT = ["input", "output"]
    MOONDREAM_PATH = "./moondream-0_5b-int8.mf"

    # Initialize processor
    start_time = time.time()
    processor = DatasetProcessor(SOURCE_REPO, TARGET_REPO, TOKEN, MOONDREAM_PATH)
    logger.info(f"Processor initialization took {time.time() - start_time:.2f} seconds")

    try:
        # Load dataset
        start_time = time.time()
        with tqdm(total=1, desc="Loading dataset") as pbar:
            dataset = processor.load_dataset()
            pbar.update(1)
        logger.info(f"Dataset loading took {time.time() - start_time:.2f} seconds")

        # Extract desired columns
        start_time = time.time()
        with tqdm(total=1, desc="Extracting columns") as pbar:
            dataset = processor.extract_columns(dataset, COLUMNS_TO_EXTRACT)
            pbar.update(1)
        logger.info(f"Column extraction took {time.time() - start_time:.2f} seconds")

        # Add new column with Moondream captions
        start_time = time.time()
        dataset = processor.add_custom_column(
            dataset,
            new_column_name="moondream_caption",
            processing_func=processor.custom_processing
        )
        logger.info(f"Caption generation took {time.time() - start_time:.2f} seconds")

        # Push to HuggingFace Hub
        processor.create_and_push_dataset(dataset)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

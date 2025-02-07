import os
import time
from functools import wraps
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, login
import logging
import io
from PIL import Image
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

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
    def __init__(self, source_repo: str, target_repo: str, token: str):
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.token = token
        self.api = HfApi()
        self.function_times = {}
        self.processed_count = 0

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
        login(token=token)

    @timing_decorator
    def load_dataset(self) -> Dataset:
        try:
            logger.info(f"Loading dataset from {self.source_repo}")
            dataset = load_dataset(self.source_repo)
            logger.info(f"Successfully loaded dataset with {len(dataset['train'])} rows")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @timing_decorator
    def generate_captions(self, png_data: bytes) -> tuple:
        """Generate both short and normal captions for an image."""
        try:
            image = Image.open(io.BytesIO(png_data))
            short_caption = self.model.caption(image, length="short")["caption"]
            normal_caption = self.model.caption(image)["caption"]
            return short_caption, normal_caption
        except Exception as e:
            logger.error(f"Error generating captions: {e}")
            return "Error generating short caption", "Error generating normal caption"

    def process_row(self, row):
        """Process a single row of the dataset."""
        try:
            self.processed_count += 1
            start_time = time.time()

            if not row['png_processed'] or row['png_data'] is None:
                return {
                    'moondream_short_caption': "Failed - No valid PNG data",
                    'moondream_normal_caption': "Failed - No valid PNG data"
                }

            short_caption, normal_caption = self.generate_captions(row['png_data'])

            processing_time = time.time() - start_time

            # Log timing information every 10 items
            if self.processed_count % 10 == 0:
                avg_times = {
                    func: sum(times)/len(times)
                    for func, times in self.function_times.items()
                }
                logger.info(
                    f"Processed {self.processed_count} items\n"
                    f"Average times:\n"
                    f"  generate_captions: {avg_times.get('generate_captions', 0):.2f}s\n"
                    f"  Total per item: {processing_time:.2f}s"
                )

            return {
                'moondream_short_caption': short_caption,
                'moondream_normal_caption': normal_caption
            }
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return {
                'moondream_short_caption': "Error processing image",
                'moondream_normal_caption': "Error processing image"
            }

    @timing_decorator
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Add caption columns to the dataset."""
        try:
            logger.info("Adding caption columns")
            total_rows = len(dataset)

            pbar = tqdm(total=total_rows, desc="Generating captions")

            def map_function(example):
                start_time = time.time()
                captions = self.process_row(example)
                example.update(captions)

                processing_time = time.time() - start_time
                pbar.set_postfix({
                    'caption_avg': f"{sum(self.function_times.get('generate_captions', [0]))/max(len(self.function_times.get('generate_captions', [1])), 1):.2f}s",
                    'total_avg': f"{processing_time:.2f}s"
                })
                pbar.update(1)
                return example

            processed_dataset = dataset.map(
                map_function,
                batched=False,
                num_proc=1  # Single process due to GPU usage
            )

            pbar.close()
            return processed_dataset

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

    @timing_decorator
    def push_to_hub(self, dataset: Dataset) -> None:
        """Push the processed dataset to HuggingFace Hub."""
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
    SOURCE_REPO = "thesantatitan/svg-rendered"
    TARGET_REPO = "thesantatitan/svg-500k-moondream-captions"
    TOKEN = os.getenv("HF_TOKEN")

    start_time = time.time()
    captioner = MoondreamCaptioner(SOURCE_REPO, TARGET_REPO, TOKEN)
    logger.info(f"Initialization took {time.time() - start_time:.2f} seconds")

    try:
        # Load dataset
        dataset = captioner.load_dataset()

        # Process dataset
        processed_dataset = captioner.process_dataset(dataset)

        # Push to HuggingFace Hub
        captioner.push_to_hub(processed_dataset)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

import os
import time
from functools import wraps
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, login
import logging
from PIL import Image
import io
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import concurrent.futures

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

class DatasetCaptioner:
    def __init__(self, source_repo: str, target_repo: str, token: str, batch_size: int = 8, num_models: int = 4):
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.token = token
        self.api = HfApi()
        self.function_times = {}
        self.processed_count = 0
        self.batch_size = batch_size
        self.num_models = num_models

        # Initialize multiple Moondream models
        logger.info(f"Initializing {num_models} Moondream models...")
        start_time = time.time()
        self.models = []
        for i in range(num_models):
            model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                trust_remote_code=True,
                device_map={"": f"cuda:{i % torch.cuda.device_count()}"}
            )
            self.models.append(model)
        logger.info(f"Moondream initialization took {time.time() - start_time:.2f} seconds")

        login(token=token)

    def generate_caption_single(self, model_idx: int, image: Image) -> str:
        """Generate caption for a single image using specified model."""
        try:
            with torch.no_grad():
                caption = self.models[model_idx].caption(image, length="short")["caption"]
                return caption
        except Exception as e:
            logger.error(f"Error generating caption with model {model_idx}: {e}")
            return "Error generating caption"

    @timing_decorator
    def generate_captions_batch(self, images: list) -> list:
        """Generate captions for a batch of images using multiple models in parallel."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_models) as executor:
                future_to_image = {
                    executor.submit(
                        self.generate_caption_single,
                        idx % self.num_models,
                        image
                    ): idx
                    for idx, image in enumerate(images)
                }

                results = [""] * len(images)
                for future in concurrent.futures.as_completed(future_to_image):
                    idx = future_to_image[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing image {idx}: {e}")
                        results[idx] = "Error generating caption"

                return results
        except Exception as e:
            logger.error(f"Error generating captions batch: {e}")
            return ["Error generating caption"] * len(images)

    def process_batch(self, examples):
        """Process a batch of images and generate captions."""
        try:
            start_time = time.time()

            # Convert binary data to PIL Images
            images = []
            for example in examples['png_data']:
                try:
                    image = Image.open(io.BytesIO(example))
                    images.append(image)
                except Exception as e:
                    logger.error(f"Error processing image in batch: {e}")
                    images.append(None)

            # Filter out None values and their corresponding indices
            valid_images = []
            valid_indices = []
            for idx, img in enumerate(images):
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(idx)

            # Generate captions for valid images
            if valid_images:
                captions = self.generate_captions_batch(valid_images)
            else:
                captions = []

            # Prepare results for all examples
            results = ["Error processing image"] * len(examples['png_data'])
            for idx, caption in zip(valid_indices, captions):
                results[idx] = caption

            self.processed_count += len(examples['png_data'])

            # Log timing information
            if self.processed_count % (self.batch_size * 5) == 0:
                avg_times = {
                    func: sum(times)/len(times)
                    for func, times in self.function_times.items()
                }
                logger.info(
                    f"Processed {self.processed_count} items\n"
                    f"Average times:\n"
                    f"  generate_captions_batch: {avg_times.get('generate_captions_batch', 0):.2f}s\n"
                    f"  Total batch time: {time.time() - start_time:.2f}s"
                )

            return {"moondream_caption": results}
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {"moondream_caption": ["Error processing batch"] * len(examples['png_data'])}

    @timing_decorator
    def load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace."""
        try:
            logger.info(f"Loading dataset from {self.source_repo}")
            dataset = load_dataset(self.source_repo, split="train")
            logger.info(f"Original dataset size: {len(dataset)} rows")

            # Filter out rows where png_processed is False
            dataset = dataset.filter(lambda x: x['png_processed'])
            logger.info(f"Dataset size after filtering: {len(dataset)} rows")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @timing_decorator
    def add_captions(self, dataset: Dataset) -> Dataset:
        """Add captions to dataset."""
        try:
            logger.info("Adding captions to dataset")
            total_rows = len(dataset)

            pbar = tqdm(total=total_rows, desc="Generating captions")

            processed_dataset = dataset.map(
                self.process_batch,
                batched=True,
                batch_size=self.batch_size,
                remove_columns=dataset.column_names
            )

            pbar.update(total_rows)
            pbar.close()
            return processed_dataset

        except Exception as e:
            logger.error(f"Error adding captions: {e}")
            raise

    @timing_decorator
    def push_dataset(self, dataset: Dataset) -> None:
        """Push processed dataset to HuggingFace Hub."""
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
    BATCH_SIZE = 32  # Adjust based on your GPU memory
    NUM_MODELS = 4  # Adjust based on number of GPUs and memory

    start_time = time.time()
    captioner = DatasetCaptioner(SOURCE_REPO, TARGET_REPO, TOKEN, BATCH_SIZE, NUM_MODELS)
    logger.info(f"Captioner initialization took {time.time() - start_time:.2f} seconds")

    try:
        # Load dataset
        dataset = captioner.load_dataset()

        # Add captions
        dataset = captioner.add_captions(dataset)

        # Push to HuggingFace Hub
        captioner.push_dataset(dataset)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

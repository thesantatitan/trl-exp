import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, login
from typing import List, Dict, Any
import logging
import cairosvg
import io
from PIL import Image
import moondream as md
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, source_repo: str, target_repo: str, token: str, moondream_path: str):
        """
        Initialize the dataset processor.

        Args:
            source_repo (str): Source repository
            target_repo (str): Target repository name
            token (str): HuggingFace API token
            moondream_path (str): Path to Moondream model file
        """
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.token = token
        self.api = HfApi()

        # Initialize Moondream model
        logger.info("Initializing Moondream model...")
        self.model = md.vl(model=moondream_path)

        # Login to Hugging Face
        login(token=token)

    def svg_to_png(self, svg_string: str) -> Image.Image:
        """Convert SVG string to PNG image."""
        try:
            # Convert SVG to PNG using CairoSVG
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))

            # Convert to PIL Image
            image = Image.open(io.BytesIO(png_data))
            return image
        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {e}")
            raise

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image using Moondream."""
        try:
            # Encode image
            encoded_image = self.model.encode_image(image)

            # Generate caption
            caption = self.model.caption(encoded_image)["caption"]
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            raise

    def custom_processing(self, row: Dict[str, Any]) -> Any:
        """
        Process SVG input:
        1. Convert SVG to PNG
        2. Generate caption using Moondream

        Args:
            row: Dataset row containing SVG in 'input' field
        Returns:
            Generated caption
        """
        try:
            # Get SVG string from input
            svg_string = row['input']

            # Convert SVG to PNG
            image = self.svg_to_png(svg_string)

            # Generate caption
            caption = self.generate_caption(image)

            return caption
        except Exception as e:
            logger.error(f"Error in custom processing: {e}")
            return "Error processing image"

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

    def extract_columns(self, dataset: Dataset, columns: List[str]) -> Dataset:
        """Extract specific columns from dataset."""
        try:
            logger.info(f"Extracting columns: {columns}")
            return dataset.select_columns(columns)
        except Exception as e:
            logger.error(f"Error extracting columns: {e}")
            raise

    def add_custom_column(self, dataset: Dataset,
                         new_column_name: str,
                         processing_func: callable) -> Dataset:
        """
        Add a new column based on existing data.

        Args:
            dataset: Input dataset
            new_column_name: Name for the new column
            processing_func: Function that takes a row and returns the new column value
        """
        try:
            logger.info(f"Adding new column: {new_column_name}")
            total_rows = len(dataset)

            # Initialize progress bar
            pbar = tqdm(total=total_rows, desc="Processing images")

            def map_function(example):
                example[new_column_name] = processing_func(example)
                pbar.update(1)
                return example

            # Use batched=False to process one example at a time and update progress bar
            processed_dataset = dataset.map(
                map_function,
                batched=False,
                num_proc=1  # Required for accurate progress bar
            )

            pbar.close()
            return processed_dataset

        except Exception as e:
            logger.error(f"Error adding custom column: {e}")
            raise

    def create_and_push_dataset(self, dataset: Dataset) -> None:
        """Create new repo and push processed dataset."""
        try:
            # Create new repository if it doesn't exist
            try:
                create_repo(self.target_repo, private=False, token=self.token)
                logger.info(f"Created new repository: {self.target_repo}")
            except Exception as e:
                logger.warning(f"Repository creation warning (might already exist): {e}")

            # Add progress bar for dataset upload
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
    TARGET_REPO = "thesantatitan/svg-instruct-moondream"
    TOKEN = os.getenv("HF_TOKEN")
    COLUMNS_TO_EXTRACT = ["input", "output"]
    MOONDREAM_PATH = "./moondream-0_5b-int8.mf"  # Update with your model path

    # Initialize processor
    processor = DatasetProcessor(SOURCE_REPO, TARGET_REPO, TOKEN, MOONDREAM_PATH)

    try:
        # Load dataset
        with tqdm(total=1, desc="Loading dataset") as pbar:
            dataset = processor.load_dataset()
            pbar.update(1)

        # Extract desired columns
        with tqdm(total=1, desc="Extracting columns") as pbar:
            dataset = processor.extract_columns(dataset, COLUMNS_TO_EXTRACT)
            pbar.update(1)

        # Add new column with Moondream captions
        dataset = processor.add_custom_column(
            dataset,
            new_column_name="moondream_caption",
            processing_func=processor.custom_processing
        )

        # Push to HuggingFace Hub
        processor.create_and_push_dataset(dataset)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

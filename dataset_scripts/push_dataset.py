import os
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_from_disk
from huggingface_hub import HfApi, create_repo

def combine_and_upload_datasets(
    base_dir: str,
    repo_name: str,
    private: bool = False,
    token: str = None,
):
    """
    Combines multiple Arrow datasets from subdirectories and uploads to Hugging Face.

    Args:
        base_dir (str): Path to the base directory containing dataset folders
        repo_name (str): Name for the Hugging Face dataset repository
        private (bool): Whether to create a private repository
        token (str): Hugging Face API token
    """
    # Collect all dataset paths
    dataset_paths = []
    base_path = Path(base_dir)

    for folder in base_path.iterdir():
        if folder.is_dir():
            dataset_paths.append(str(folder))

    if not dataset_paths:
        raise ValueError(f"No dataset directories found in {base_dir}")

    print(f"Found {len(dataset_paths)} datasets to combine")

    # Load and combine datasets
    datasets = []
    for path in dataset_paths:
        try:
            dataset = load_from_disk(path)
            datasets.append(dataset)
            print(f"Successfully loaded dataset from {path}")
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets were successfully loaded")

    # Combine all datasets
    combined_dataset = concatenate_datasets(datasets)
    print(f"Combined dataset has {len(combined_dataset)} rows")

    # Create HF repository if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            token=token
        )
        print(f"Created new repository: {repo_name}")
    except Exception as e:
        print(f"Repository {repo_name} might already exist: {e}")

    # Push to Hugging Face
    combined_dataset.push_to_hub(
        repo_id=repo_name,
        token=token,
        private=private
    )
    print(f"Successfully pushed dataset to {repo_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine and upload Arrow datasets to Hugging Face")
    parser.add_argument("--base_dir", required=True, help="Base directory containing dataset folders")
    parser.add_argument("--repo_name", required=True, help="Name for the Hugging Face dataset repository")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--token", required=True, help="Hugging Face API token")

    args = parser.parse_args()

    combine_and_upload_datasets(
        base_dir=args.base_dir,
        repo_name=args.repo_name,
        private=args.private,
        token=args.token
    )

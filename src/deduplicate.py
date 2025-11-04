import datetime
import gc
import logging

from huggingface_hub import HfFolder, login

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loaders.utils import parse
from src.logger import setup_logger

logger = logging.getLogger(__name__)


def post_process(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Post-process the dataset to remove duplicate entries based on the "text" field, and clean up unnecessary columns.

    Parameters
    ----------
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset
        The input dataset to be deduplicated.

    Returns
    -------
    DatasetDict | Dataset | IterableDatasetDict | IterableDataset
        The deduplicated dataset with duplicate entries removed.
    """
    seen_hashes = set()

    def is_unique_text(example):
        text_hash = hash(example["text"])
        if text_hash in seen_hashes:
            return False
        seen_hashes.add(text_hash)
        return True

    unique_data = dataset.filter(is_unique_text)

    return unique_data.remove_columns(["source", "subset", "source_split"])


def main():
    """Main function to deduplicate a dataset and push the cleaned version to the Hugging Face Hub."""
    args = parse()

    # Create logger instance
    if args.log_level == "DEBUG":
        log_level = logging.DEBUG
    elif args.log_level == "INFO":
        log_level = logging.INFO
    else:
        raise ValueError(args.log_level)
    setup_logger(log_level)

    # Reading HuggingFace token and logging in to pull the dataset
    with open(args.hf_token, "r") as f:
        hf_token = f.read().strip()
    login(token=hf_token)
    HfFolder.save_token(hf_token)

    logger.info(
        "Loading dataset from %s" % "LIMICS/PARTAGES-sourced"
        if args.make_commercial_version
        else "LIMICS/PARTAGES-Research-sourced"
    )

    new_dataset = None

    # Loading the sourced dataset and processing it to the deduplicated using a context manager for better resource efficiency
    with load_dataset(
        "parquet",
        data_files="hf://datasets/LIMICS/PARTAGES-sourced/*/*.parquet"
        if args.make_commercial_version
        else "hf://datasets/LIMICS/PARTAGES-Research-sourced/*/*.parquet",
        split="train",
        download_mode="force_redownload",
    ) as raw_dataset:  # type: ignore
        logger.info(f"Shape of raw dataset: {raw_dataset.shape}")  # type: ignore
        new_dataset = post_process(dataset=raw_dataset)
        new_dataset = new_dataset.shuffle(seed=42)
        new_dataset = new_dataset.flatten_indices(keep_in_memory=False)  # type: ignore
        logger.info(f"Shape of deduplicated dataset: {new_dataset.shape}")  # type: ignore
    gc.collect()

    logger.info(
        "Writing deduplicated dataset to %s" % "LIMICS/PARTAGES"
        if args.make_commercial_version
        else "LIMICS/PARTAGES-Research"
    )

    # Writing the deduplicated dataset to the corresponding HuggingFace repository
    new_dataset.push_to_hub(  # type: ignore
        repo_id="LIMICS/PARTAGES"
        if args.make_commercial_version
        else "LIMICS/PARTAGES-Research",
        commit_message="Update dataset on %s" % datetime.date.today().isoformat(),
    )


if __name__ == "__main__":
    main()

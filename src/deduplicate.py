import datetime
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
    seen_hashes = set()

    def is_unique_text(example):
        text_hash = hash(example["text"])
        if text_hash in seen_hashes:
            return False
        seen_hashes.add(text_hash)
        return True

    unique_data = dataset.filter(is_unique_text)

    return unique_data


def main():
    args = parse()

    if args.log_level == "DEBUG":
        log_level = logging.DEBUG
    elif args.log_level == "INFO":
        log_level = logging.INFO
    else:
        raise ValueError(args.log_level)
    # Create logger
    setup_logger(log_level)

    # Reading HuggingFace token
    with open(args.hf_token, "r") as f:
        hf_token = f.read().strip()
    # Login to Hugging Face Hub
    login(token=hf_token)
    HfFolder.save_token(hf_token)

    logger.info(
        "Loading dataset from %s" % "LIMICS/PARTAGES-sourced"
        if args.make_commercial_version
        else "LIMICS/PARTAGES-Research-sourced"
    )

    raw_dataset = load_dataset(
        "parquet",
        data_files="hf://datasets/LIMICS/PARTAGES-sourced/*/*.parquet"
        if args.make_commercial_version
        else "hf://datasets/LIMICS/PARTAGES-Research-sourced/*/*.parquet",
        split="train",
        download_mode="force_redownload",
    )

    logger.info(f"Shape of raw dataset: {raw_dataset.shape}")  # type: ignore
    new_dataset = post_process(dataset=raw_dataset)
    logger.info(f"Shape of deduplicated dataset: {new_dataset.shape}")  # type: ignore

    logger.info(
        "Writing deduplicated dataset to %s" % "LIMICS/PARTAGES"
        if args.make_commercial_version
        else "LIMICS/PARTAGES-Research"
    )

    new_dataset.push_to_hub(  # type: ignore
        repo_id="LIMICS/PARTAGES"
        if args.make_commercial_version
        else "LIMICS/PARTAGES-Research",
        commit_message="Update dataset on %s" % datetime.date.today().isoformat(),
    )


if __name__ == "__main__":
    main()

import datetime
import gc
import logging
from multiprocessing import Pool, cpu_count

from datasketch import MinHash, MinHashLSH
from huggingface_hub import HfFolder, login

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loaders.utils import parse, read_config
from src.logger import setup_logger

logger = logging.getLogger(__name__)

NUM_PERM = 128
SIMILARITY_THRESHOLD = 0.85
NUM_WORKERS = cpu_count() - 2


def text_to_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for token in text.split():
        m.update(token.encode("utf8"))
    return m


def filter_hashes(dataset: list, minhash_list):
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERM)
    unique_rows = {"instruction": [], "input": [], "output": [], "source": []}
    keys = set()
    indexes = []
    idx = 0
    for ex, mh in zip(dataset, minhash_list):
        matches = lsh.query(mh)
        if len(matches) == 0:
            key = f"{idx}"
            lsh.insert(key, mh)
            unique_rows["instruction"].append(ex["instruction"])
            unique_rows["input"].append(ex["input"])
            unique_rows["output"].append(ex["output"])
            unique_rows["source"].append(ex["source"])
            indexes.append(idx)
            keys.add(key)
        idx += 1
    return unique_rows


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
        text_hash = hash(example["input"])
        if text_hash in seen_hashes:
            return False
        seen_hashes.add(text_hash)
        return True

    unique_data = dataset.filter(is_unique_text)
    new_dataset = unique_data.remove_columns(["data_dir", "source_split"])
    logger.info(unique_data)
    logger.info(new_dataset)

    with Pool(NUM_WORKERS) as p:
        minhashes = p.map(text_to_minhash, new_dataset["input"])  # type: ignore

    deduped_data = filter_hashes(
        dataset=new_dataset,  # type: ignore
        minhash_list=minhashes,
    )

    return Dataset.from_dict(deduped_data)


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

    deduped = None

    all_configs = read_config()
    all_configs = [
        cfg for cfg in all_configs if cfg["adaptation_type"] == args.adaptation_type
    ]

    if args.make_commercial_version:
        data_files = [
            "hf://datasets/LIMICS/PARTAGES-sourced/%s/%s.parquet"
            % (cfg["source"], cfg["source"])
            for cfg in [
                cfg
                for cfg in all_configs
                if cfg["commercial_use"] == args.make_commercial_version
            ]
        ]
    else:
        data_files = [
            "hf://datasets/LIMICS/PARTAGES-Research-sourced/%s/%s.parquet"
            % (cfg["source"], cfg["source"])
            for cfg in all_configs
        ]

    if data_files:
        # Loading the sourced dataset and processing it to the deduplicated using a context manager for better resource efficiency
        with load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            download_mode="force_redownload",
        ) as raw_dataset:  # type: ignore
            logger.info(f"Shape of raw dataset: {raw_dataset.shape}")  # type: ignore
            if args.adaptation_type == "instruction-tuning":
                logger.info(
                    "Adaptiation %s: keeping raw data." % (args.adaptation_type)
                )
                deduped = raw_dataset
            else:
                logger.info(
                    "Adaptiation %s: deduplicating raw data." % (args.adaptation_type)
                )
                deduped = post_process(dataset=raw_dataset)
            deduped = deduped.shuffle(seed=42)
            deduped = deduped.flatten_indices(keep_in_memory=False)  # type: ignore
            logger.info(f"Shape of deduplicated dataset: {deduped.shape}")  # type: ignore
        gc.collect()

        # Writing the deduplicated dataset to the corresponding HuggingFace repository
        if args.push_to_hub:
            logger.info(
                "Writing deduplicated dataset to %s into data_dir %s."
                % (
                    "LIMICS/PARTAGES"
                    if args.make_commercial_version
                    else "LIMICS/PARTAGES-Research",
                    args.adaptation_type,
                )
            )
            deduped.push_to_hub(  # type: ignore
                repo_id="LIMICS/PARTAGES"
                if args.make_commercial_version
                else "LIMICS/PARTAGES-Research",
                split="train",  # type: ignore
                data_dir=args.adaptation_type,
                commit_message="Update %s dataset on %s"
                % (args.adaptation_type, datetime.date.today().isoformat()),
            )
    else:
        logger.info("")  # TODO
        pass


if __name__ == "__main__":
    main()

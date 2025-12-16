import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfFolder, login

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loaders.utils import (
    compute_dataset_stats,
    compute_global_stats,
    read_adaptation_type,
    read_config,
    str2bool,
)
from src.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--hf_token", type=str, help="HuggingFace API token file path.")
    parser.add_argument("--is_sourced", type=str2bool, default=True)
    parser.add_argument("--make_commercial_version", type=str2bool, default=True)
    parser.add_argument(
        "--adaptation_type", type=read_adaptation_type, default="fine-tuning"
    )
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def load_sourced_dataset(make_commercial, adaptation_type):
    """
    Load sourced dataset based on commercial use and adaptation type.

    Parameters
    ----------
    make_commercial : bool
        Whether to load the commercial version of the dataset.
    adaptation_type : str
        The adaptation type for the dataset.

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset.
    """
    all_configs = read_config()
    all_configs = [
        cfg for cfg in all_configs if cfg["adaptation_type"] == adaptation_type
    ]
    if make_commercial:
        data_files = [
            "hf://datasets/LIMICS/PARTAGES-sourced/%s/%s.parquet"
            % (cfg["source"], cfg["source"])
            for cfg in [
                cfg for cfg in all_configs if cfg["commercial_use"] == make_commercial
            ]
        ]
    else:
        data_files = [
            "hf://datasets/LIMICS/PARTAGES-Research-sourced/%s/%s.parquet"
            % (cfg["source"], cfg["source"])
            for cfg in all_configs
        ]
    return load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        download_mode="force_redownload",
    )


def choose_load_dataset(
    is_sourced: bool = True,
    make_commercial: bool = True,
    adaptation_type: str = "fine-tuning",
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """
    Choose and load the appropriate dataset based on parameters.

    Parameters
    ----------
    is_sourced : bool, optional
        Whether to load a sourced dataset, by default True.
    make_commercial : bool, optional
        Whether to load the commercial version of the dataset, by default True.
    adaptation_type : str, optional
        The adaptation type for the dataset, by default "fine-tuning".

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset.
    """
    if is_sourced:
        return load_sourced_dataset(
            make_commercial=make_commercial, adaptation_type=adaptation_type
        )
    else:
        return load_dataset(
            "LIMICS/PARTAGES" if make_commercial else "LIMICS/PARTAGES-Research",
            split="train",
            # data_dir=adaptation_type,
            download_mode="force_redownload",
        )


def main():
    """
    Main function to compute and save dataset statistics.
    """
    args = parse_args()

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

    dataset = choose_load_dataset(
        is_sourced=args.is_sourced,
        make_commercial=args.make_commercial_version,
        adaptation_type=args.adaptation_type,
    )
    if args.is_sourced:
        stats = {}
        dataset_names = list(np.unique(dataset["source"]))  # type: ignore
        for name in dataset_names:
            stats[name] = compute_dataset_stats(
                dataset=dataset.filter(lambda example: example["source"] == name),
                source_name=name,
            )
        df = pd.DataFrame(list(stats.values()), index=list(stats.keys()))
        compute_global_stats(df=df)
        df.to_csv(args.save_path, index=True)
    else:
        stats = {}
        stats["globals"] = compute_dataset_stats(dataset=dataset, source_name="")
        df = pd.DataFrame(list(stats.values()), list(stats.keys()))
        df.to_csv(args.save_path, index=True)


if __name__ == main():
    main()

import argparse
import datetime
import gzip
import io
import logging
import os
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import pandas as pd
import yaml

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    Value,
    arrow_dataset,
)
from src.text_cleaning import cleaner

logger = logging.getLogger(__name__)


def str2bool(v: str) -> bool:
    """
    Convert a string to a boolean value.

    Parameters
    ----------
    v : str
        Input string ("yes", "true", "t", "1" are all converted to True, otherwise False).

    Returns
    -------
    bool
        Corresponding boolean value.
    """
    return v.lower() in ("yes", "true", "t", "1")


def parse() -> argparse.Namespace:
    """
    Parse command-line arguments for script configuration.

    Returns
    -------
    argparse.Namespace
        Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token", type=str, default="", help="HuggingFace API token file path."
    )
    parser.add_argument(
        "--push_to_hub",
        type=str2bool,
        default=False,
        help="Whether to push dataset to HF hub.",
    )
    parser.add_argument(
        "--use_all_sources",
        type=str2bool,
        default=True,
        help="Wether to use all sources in config.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Name of the source to use (if not using all sources).",
    )
    parser.add_argument(
        "--make_commercial_version",
        type=str2bool,
        default=True,
        help="Whether to filter out non-commercial use datasets.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level for the script.",
    )
    return parser.parse_args()


def read_config(path: Union[Path, str] = "config/datasets.yaml") -> dict:
    """
    Read the YAML configuration file for datasets.

    Parameters
    ----------
    path : Union[Path, str], optional
        Path to the configuration file.

    Returns
    -------
    dict
        Dictionary of dataset configurations.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]


def read_compressed(path: Union[Path, str]) -> Generator[str, None, None]:
    """
    Read a compressed text file in .gz format and return its lines.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the directory containing the compressed file.

    Returns
    -------
    list
        List of lines from the decompressed file.
    """
    all_bytes = b""
    for part in sorted(os.listdir(path=path)):
        with open(Path(path) / part, "rb") as f:
            all_bytes += f.read()
    with gzip.open(io.BytesIO(all_bytes), "rt", encoding="utf-8") as res:
        for line in res:
            yield line.rstrip("\n")


def generate_info_file(
    dataset: arrow_dataset.Dataset,
    source_name: str,
    source_split: str,
    comment: str,
    stats: dict,
) -> str:
    """
    Generate a markdown-formatted information file for the dataset.

    Parameters
    ----------
    dataset : arrow_dataset.Dataset
        The dataset object.
    source_name : str
        The name of the data source.
    source_split : str
        The split of the data source (e.g., "train", "validation").
    comment : str
        Additional comments about the dataset.
    stats : dict
        Statistics about the dataset.

    Returns
    -------
    str
        The formatted information file content.
    """
    return (
        f"# {source_name} \n"
        f"## Presentation \n{comment} \n"
        f"## Version \nDate of latest push: {datetime.date.today().isoformat()} \n"
        f"## Splits \n{source_split} \n"
        f"## Architecture and shape \n{dataset} \n"
        f"Shape: {dataset.shape}"
        f"## Stats \n"
        f"{pd.DataFrame(list(stats.values()), index=list(stats.keys())).to_string(index=False, float_format='{:.2f}'.format)}\n"
    )


def load_config(args: argparse.Namespace) -> list:
    """
    Load dataset configuration based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    list
        List of dataset configurations.
    """
    all_cfg = read_config()
    if isinstance(all_cfg, dict):
        all_cfg = [all_cfg]

    if not args.use_all_sources:
        for cfg in all_cfg:
            if args.source == cfg["source"]:
                all_cfg = [cfg]
                break
        else:
            raise ValueError(f"No available dataset named {args.source} in config.")

    if args.make_commercial_version:
        logger.info("COMMERCIAL VERSION")
        logger.info(
            f"Available datasets in config: {[cfg['source'] for cfg in all_cfg]}"
        )
        tmp_cfg = []
        for cfg in all_cfg:
            if cfg["commercial_use"]:
                tmp_cfg.append(cfg)
        all_cfg = tmp_cfg
        logger.info(
            f"Remaining datasets after commercial use filtering: {[cfg['source'] for cfg in all_cfg]}"
        )
    else:
        logger.info("NON-COMMERCIAL VERSION")
        logger.info(
            f"Available datasets in config: {[cfg['source'] for cfg in all_cfg]}"
        )

    if len(all_cfg) < 1:
        raise RuntimeError(
            "No available dataset(s) for given parametrization (check commercial use and source(s) given)."
        )

    return all_cfg


def load_local(
    path: Union[str, Path],
    split: Union[str, list],
    data_dir: Optional[Union[str, list]] = None,
    streaming: bool = False,
    trust_remote_code: bool = True,
) -> dict[str, IterableDataset] | IterableDataset | Dataset | DatasetDict:
    """
    Load a local dataset from a specified path, handling both .txt and .gz files.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory containing the dataset files.
    split : Union[str, list]
        The data split to load (e.g., "train", "validation", "test").
    data_dir : Union[str, list], optional
        Subdirectories within the path to look for data files.
    streaming : bool, optional
        Whether to load the dataset in streaming mode (not implemented).
    trust_remote_code : bool, optional
        Whether to trust remote code when loading the dataset (not implemented).

    Returns
    -------
    Dataset
        The loaded dataset.
    """
    logger.info(f"Loading from local path: {path} for split: {split}")
    path = Path(path)

    def iter_gz(path: Path):
        all_bytes = b""
        for part in sorted(os.listdir(path)):
            with open(path / part, "rb") as f:
                all_bytes += f.read()
        with gzip.open(io.BytesIO(all_bytes), "rt", encoding="utf-8") as res:
            for line in res:
                yield {"text": line.rstrip("\n")}

    def iter_txt(path: Path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".txt"):
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", encoding="utf-8") as fh:
                        yield {"text": fh.read()}

    all_files = [p for p in path.rglob("*") if p.is_file()]
    has_gz = any(p.suffix == ".gz" for p in all_files)
    has_txt = any(p.suffix == ".txt" for p in all_files)
    if has_gz:
        return Dataset.from_generator(lambda: iter_gz(path))
    elif has_txt:
        return Dataset.from_generator(lambda: iter_txt(path))
    else:
        raise RuntimeError(
            f"No .txt or .gz files found in {path} or its subdirectories."
        )


def compute_dataset_stats(
    dataset: Dataset,
    source_name: str,
    subset: str,
    split: str,
) -> dict:
    """
    Compute basic statistics for a text dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object.
    source_name : str
        The name of the data source.
    subset : str
        The subset of the data source.
    split : str
        The split of the data source (e.g., "train", "validation").

    Returns
    -------
    dict
        Dictionary containing dataset statistics.
    """
    if "text" in dataset.column_names:
        words_lens = [len(txt.split()) for txt in dataset["text"]]
        chars_lens = [len(txt) for txt in dataset["text"]]
        row = {
            "nb_docs": dataset.shape[0],
            "nb_words": sum(words_lens),
            "mean_words": np.mean(words_lens),
            "std_words": np.std(words_lens, ddof=0),
            "nb_chars": sum(chars_lens),
            "mean_chars": np.mean(chars_lens),
            "std_chars": np.std(chars_lens, ddof=0),
        }
        return row
    else:
        raise ValueError(
            f'Dataset "{source_name}" does not contain a \'text\' column for subset "{subset}" and split "{split}".'
        )


def pooled_variance(mode: str, row: dict, avg: float) -> float:
    """
    Compute pooled variance for multiple subsets i: σ² = Σ(nᵢ(σᵢ² + (μᵢ - μ)²)) / (Σ nᵢ).

    Parameters
    ----------
    mode : str
        Mode of variance to compute ("word" or "char").
    row : pd.Series
        Row containing statistics for a specific subset.
    avg : float
        Overall mean for the pooled variance calculation.

    Returns
    -------
    float
        The computed pooled variance.
    """
    if mode == "word":
        return row["nb_docs"] * (row["std_words"] ** 2 + (row["mean_words"] - avg) ** 2)
    elif mode == "char":
        return row["nb_docs"] * (row["std_chars"] ** 2 + (row["mean_chars"] - avg) ** 2)
    else:
        raise ValueError(f"Unkown mode option for pooled variance computation: {mode}.")


def weighted_avg_variance(
    mode: str,
    df: pd.DataFrame,
    total: float,
) -> float:
    """
    Commpute weighted average variance for multiple subsets i: σ² = (Σ nᵢσᵢ²) / (Σ nᵢ).

    Parameters
    ----------
    mode : str
        Mode of variance to compute ("word" or "char").
    df : pd.DataFrame
        DataFrame containing statistics for multiple subsets.
    total : float
        Total number of documents across all subsets.

    Returns
    -------
    float
        The computed weighted average variance.
    """
    if mode == "word":
        return np.sqrt(
            np.sum(
                [
                    df["nb_docs"].iloc[i] * df["std_words"].iloc[i] ** 2
                    for i in range(len(df))
                ]
            )
            / total
        )
    elif mode == "char":
        return np.sqrt(
            np.sum(
                [
                    df["nb_docs"].iloc[i] * df["std_chars"].iloc[i] ** 2
                    for i in range(len(df))
                ]
            )
            / total
        )
    else:
        raise ValueError(
            f"Unkown mode option for weighted average variance computation: {mode}."
        )


def update_row(
    base_row: dict,
    add_row: dict,
) -> dict:
    """
    Update a statistics row by adding values from another row.

    Parameters
    ----------
    base_row : dict
        The base statistics row to be updated.
    add_row : dict
        The statistics row to add to the base row.

    Returns
    -------
    dict
        Merged row with updated statistics.
    """
    new_row = {
        "nb_docs": base_row["nb_docs"] + add_row["nb_docs"],
        "nb_words": base_row["nb_words"] + add_row["nb_words"],
        "nb_chars": base_row["nb_chars"] + add_row["nb_chars"],
    }
    new_row["mean_words"] = new_row["nb_words"] / new_row["nb_docs"]
    new_row["std_words"] = np.sqrt(
        np.sum(
            [
                pooled_variance(mode="word", row=r, avg=new_row["mean_words"])
                for r in [base_row, add_row]
            ]
        )
        / new_row["nb_docs"]
    )
    new_row["mean_chars"] = new_row["nb_chars"] / new_row["nb_docs"]
    new_row["std_chars"] = np.sqrt(
        np.sum(
            [
                pooled_variance(mode="char", row=r, avg=new_row["mean_chars"])
                for r in [base_row, add_row]
            ]
        )
        / new_row["nb_docs"]
    )
    return new_row


def compute_global_stats(df: pd.DataFrame) -> None:
    """
    Compute and append global statistics to a DataFrame containing dataset statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing statistics for multiple subsets. The function modifies the DataFrame in place.

    Returns
    -------
    None
    """
    totals = {
        "nb_docs": df["nb_docs"].sum(),
        "nb_words": df["nb_words"].sum(),
        "nb_chars": df["nb_chars"].sum(),
    }
    totals["mean_words"] = totals["nb_words"] / totals["nb_docs"]
    totals["std_words"] = np.sqrt(
        np.sum(
            [
                pooled_variance(
                    mode="word", row=df.loc[source], avg=totals["mean_words"]
                )
                for source in list(df.index)
            ]
        )
        / totals["nb_docs"]
    )
    # totals['std_words'] = weighted_avg_variance(mode="word", df=df, total=totals['nb_docs'])
    totals["mean_chars"] = totals["nb_chars"] / totals["nb_docs"]
    totals["std_chars"] = np.sqrt(
        np.sum(
            [
                pooled_variance(
                    mode="char", row=df.loc[source], avg=totals["mean_chars"]
                )
                for source in list(df.index)
            ]
        )
        / totals["nb_docs"]
    )
    # totals['std_chars'] = weighted_avg_variance(mode="char", df=df, total=totals['nb_docs'])
    df.loc["Total"] = pd.Series(totals)


def clean_example(
    example: dict,
    lower: bool,
    rm_new_lines: bool,
) -> dict:
    """
    Clean the text in a dataset example using specified cleaning options.

    Parameters
    ----------
    example : dict
        A dictionary representing a dataset example with a "text" field.
    lower : bool
        Whether to convert text to lowercase.
    rm_new_lines : bool
        Whether to remove new line characters from the text.

    Returns
    -------
    dict
        The cleaned example with updated "text" field.
    """
    example["text"] = cleaner(
        example["text"], do_lower=lower, rm_new_lines=rm_new_lines
    )
    return example


def cast_columns(
    dataset: Dataset,
) -> Dataset:
    new_features = dataset.features.copy()
    for col, feature in new_features.items():
        if feature.dtype == "null":
            new_features[col] = Value("string")
    return dataset.cast(Features(new_features))

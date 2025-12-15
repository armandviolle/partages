import datetime
import logging
import os
import tempfile

from huggingface_hub import HfFolder, Repository, login

from datasets import concatenate_datasets
from loaders import REGISTRY
from loaders.utils import (  # TODO n°2 generate_info_file,
    cast_columns,
    load_config,
    parse,
    select_repo,
)
from src.logger import setup_logger

logger = logging.getLogger(__name__)


def main():
    """Main function to load, process, and optionally push datasets to the Hugging Face Hub."""

    args = parse()

    # Sanity check: log level
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

    # Load dataset configurations
    all_cfg = load_config(args=args)

    commit_files = {}
    global_ds = []

    for cfg in all_cfg:
        all_ds = []
        logger.info(f"Loading dataset {cfg['source']}: using {REGISTRY[cfg['source']]}")
        LoaderCls = REGISTRY[cfg["source"]]

        data_dirs = (
            cfg["data_dir"] if isinstance(cfg["data_dir"], list) else [cfg["data_dir"]]
        )
        splits = (
            cfg["source_split"]
            if isinstance(cfg["source_split"], list)
            else [cfg["source_split"]]
        )

        for data_dir in data_dirs:
            for split in splits:
                loader = LoaderCls(
                    source=cfg["source"],
                    path=cfg["path"],
                    data_dir=data_dir,
                    source_split=split,
                    adaptation_type=cfg["adaptation_type"],
                )
                ds = loader.load()
                ds = cast_columns(ds)
                all_ds.append(ds)
        if all_ds:
            # Concatenate all data_dirs and splits for a single source
            merged = concatenate_datasets(all_ds)
            logger.info(f"Features scheme of concatenated dataset: {merged.features}")
            logger.info(f"Shape of concatenated dataset: {merged.shape}")
            logger.debug(f"Type of concatenated datasets: {type(merged)}")
            # Remove empty rows for finetuning datasets
            if cfg["adaptation_type"] == "fine-tuning":
                df = merged.filter(lambda example: len(example["input"].strip()) > 0)
            elif cfg["adaptation_type"] == "instruction-tuning":
                df = merged.filter(
                    lambda example: len(example["instruction"].strip()) > 0
                    and len(example["input"].strip()) > 0
                    and len(example["output"].strip()) > 0
                )
            logger.info(f"Shape of concatenated dataset without empty rows: {df.shape}")
            global_ds.append(df)

            if args.push_to_hub:
                commit_files[cfg["source"]] = [df]
        else:
            raise ValueError(f'No data was loaded for dataset "{cfg["source"]}".')

    global_merged = concatenate_datasets(global_ds)
    logger.info(f"Features scheme of concatenated dataset: {global_merged.features}")
    logger.info(f"Shape of concatenated dataset: {global_merged.shape}")

    if args.push_to_hub:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(
                local_dir=tmpdir,
                clone_from=select_repo(args=args),
                repo_type="dataset",
            )
            # Dictionary commit_files is empty if not args.push_to_hub (see rows 58-60)
            for key, val in commit_files.items():
                os.makedirs(os.path.join(tmpdir, key), exist_ok=True)
                df_file_name = f"{key}/{key}.parquet"
                df_file = os.path.join(tmpdir, df_file_name)
                val[0].to_parquet(df_file)
                repo.git_add(df_file_name)

            commit_msg = (
                f"Updating dataset on {datetime.date.today().isoformat()}."
                if args.use_all_sources
                else f"Updating corpus {all_cfg[0]['source']} on {datetime.date.today().isoformat()}."
            )
            repo.git_commit(commit_message=commit_msg)
            repo.git_push()


if __name__ == "__main__":
    main()

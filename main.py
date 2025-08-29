import datetime
import json
import os
import tempfile

import pandas as pd
from huggingface_hub import HfFolder, Repository, login

from datasets import concatenate_datasets
from loaders import REGISTRY
from loaders.utils import (
    compute_dataset_stats,
    compute_global_stats,
    generate_info_file,
    load_config,
    parse,
    update_row,
)


def main():
    """Main function to load, process, and optionally push datasets to the Hugging Face Hub."""

    args = parse()
    with open(args.hf_token, "r") as f:
        hf_token = f.read().strip()
    print(hf_token)
    login(token=hf_token)
    HfFolder.save_token(hf_token)
    all_cfg = load_config(args=args)

    stats = {}
    commit_files = {}
    split_file = {"train": [], "test": []}

    for cfg in all_cfg:
        all_ds = []
        print(f"Loading dataset {cfg['source']}: using {REGISTRY[cfg['source']]}")
        LoaderCls = REGISTRY[cfg["source"]]

        subsets = cfg["subset"] if isinstance(cfg["subset"], list) else [cfg["subset"]]
        splits = (
            cfg["source_split"]
            if isinstance(cfg["source_split"], list)
            else [cfg["source_split"]]
        )

        for subset in subsets:
            for split in splits:
                try:
                    loader = LoaderCls(
                        source=cfg["source"],
                        path=cfg["path"],
                        subset=subset,
                        source_split=split,
                    )
                    ds = loader.load()
                    row = compute_dataset_stats(
                        dataset=ds,
                        source_name=cfg["source"],
                        subset=subset,
                        split=split,
                    )
                    if cfg["source"] in list(stats.keys()):
                        stats[cfg["source"]] = update_row(
                            base_row=stats[cfg["source"]], add_row=row
                        )
                    else:
                        stats[cfg["source"]] = row
                    all_ds.append(ds)
                except Exception:
                    print(f'Unavailable data split "{split}" for data_dir "{subset}".')
                    continue
        if all_ds:
            merged = concatenate_datasets(all_ds)
            print(type(merged))
            print(f"Shape of concatenated dataset: {merged.shape}")
            if args.push_to_hub:
                msg = generate_info_file(
                    dataset=merged,
                    source_name=cfg["source"],
                    source_split=cfg["source_split"],
                    comment=cfg["comment"],
                    stats=stats[cfg["source"]],
                )
                commit_files[cfg["source"]] = [msg, merged, cfg["target_split"]]
                split_file[cfg["target_split"]].append(cfg["source"])
        else:
            print(f'No data was loaded for dataset "{cfg["source"]}".')
            raise ValueError()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repository(
            local_dir=tmpdir,
            clone_from="LIMICS/PARTAGES"
            if args.make_commercial_version
            else "LIMICS/PARTAGES-research",
            repo_type="dataset",
        )
        # Dictionary commit_files is empty if not args.push_to_hub (see rows 58-60)
        for key, val in commit_files.items():
            # if not os.path.exists(os.path.join(tmpdir, key)):
            os.makedirs(os.path.join(tmpdir, key), exist_ok=True)
            # Writing and saving info file
            info_file_name = f"{key}/{key}_info.md"
            info_file = os.path.join(tmpdir, info_file_name)
            with open(info_file, "w") as f:
                f.write(val[0])
            repo.git_add(info_file_name)
            # Saving dataframe
            df_file_name = f"{key}/{key}.parquet"
            df_file = os.path.join(tmpdir, df_file_name)
            val[1].to_parquet(df_file)
            repo.git_add(df_file_name)

        stats_path = os.path.join(tmpdir, "dataset_stats.csv")
        if not args.use_all_sources and os.path.exists(stats_path):
            df = pd.read_csv(stats_path, index_col=0)
            df.drop("Total", inplace=True)
            df[args.source] = stats[args.source]
            compute_global_stats(df=df)
            df.to_csv(stats_path, index=True)
        else:
            df = pd.DataFrame(list(stats.values()), index=list(stats.keys()))
            compute_global_stats(df=df)
            df.to_csv(stats_path, index=True)

        split_path = os.path.join(tmpdir, "split.json")
        if not args.use_all_sources and os.path.exists(split_path):
            with open(split_path, "r") as f:
                old_split_file = json.load(f)
            if not (
                (args.name in old_split_file["train"])
                or (args.name in old_split_file["test"])
            ):
                old_split_file["test"].append(args.name) if (
                    args.name in split_file["test"]
                ) else old_split_file["train"].append(args.name)
            else:
                if (args.name in old_split_file["train"]) and (
                    args.name in split_file["test"]
                ):
                    old_split_file["train"].remove(args.name)
                    old_split_file["test"].append(args.name)
                elif (args.name in old_split_file["test"]) and (
                    args.name in split_file["train"]
                ):
                    old_split_file["test"].remove(args.name)
                    old_split_file["train"].append(args.name)
            with open(split_path, "w") as f:
                json.dump(old_split_file, f, indent=4)
        else:
            with open(split_path, "w") as f:
                json.dump(split_file, f, indent=4)

        if args.push_to_hub:
            repo.git_add(stats_path)
            repo.git_add(split_path)
            commit_msg = (
                f"Updating dataset on {datetime.date.today().isoformat()}."
                if args.use_all_sources
                else f"Updating corpus {all_cfg[0]['source']} on {datetime.date.today().isoformat()}."
            )
            repo.git_commit(commit_message=commit_msg)
            repo.git_push()

    with pd.option_context("display.max_columns", None, "display.width", 0):
        print(df)

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF


if __name__ == "__main__":
    main()

import os, sys
import pandas as pd
import numpy as np
from datasets import concatenate_datasets
from loaders import REGISTRY
from huggingface_hub import HfFolder, HfApi, login, Repository
import tempfile
import datetime
from pathlib import Path
from loaders.utils import parse, load_config, get_nb_characters, get_nb_words


def main():

    api = HfApi()
    args = parse()
    with open(args.hf_token, 'r') as f:
        hf_token = f.read()
    print(hf_token)
    login(token=hf_token)
    HfFolder.save_token(hf_token)
    all_cfg = load_config(args=args)

    stats = []
    global_ds = []
    info_messages = {}

    for cfg in all_cfg:
        all_ds = []
        print(f"Loading dataset {cfg['source']}: using {REGISTRY[cfg['source']]}")
        LoaderCls = REGISTRY[cfg["source"]]
    
        subsets = cfg['subset'] if isinstance(cfg['subset'], list) else [cfg['subset']]
        splits = cfg['source_split'] if isinstance(cfg['source_split'], list) else [cfg['source_split']]

        for subset in subsets: 
            for split in splits:
                try:
                    loader = LoaderCls(
                        source=cfg['source'], 
                        path=cfg['path'], 
                        subset=subset, 
                        source_split=split
                    )
                    ds = loader.load()
                    ### TODO #1 -> factorize stats computation into a utils.py function
                    print(f"Shape of {cfg['source']}-{subset}-{split}: {ds.shape}")
                    nb_chars = get_nb_characters(ds)
                    nb_words = get_nb_words(ds)
                    print(f"Number of characters = {nb_chars}")
                    print(f"Number of words = {nb_words}")
                    row = {
                        'source': cfg['source'],
                        'subset': subset,
                        'split': split,
                        'nb_chars': nb_chars,
                        'nb_words': nb_words,
                        'nb_docs': ds.shape[0],
                        "mean_words": np.mean([len(txt.split()) for txt in ds["text"]]),
                        "std_chars": np.std([len(txt) for txt in ds["text"]], ddof=0),
                        "std_words": np.std([len(txt.split()) for txt in ds["text"]], ddof=0),
                    }
                    print(f"Mean of words = {row['mean_words']}")
                    print(f"Std of characters = {row['std_chars']}")
                    print(f"Std of words = {row['std_words']}")
                    print()
                    stats.append(row)
                    # end todo
                    all_ds.append(ds)
                except Exception as e:
                    print(f"Unavailable data split \"{split}\" for data_dir \"{subset}\".")
                    continue
        if all_ds:
            merged = concatenate_datasets(all_ds)
            print(type(merged))
            print(f"Shape of concatenated dataset: {merged.shape}")
            global_ds.append(merged)
            if args.push_to_hub:
                ### TODO #2 -> modify / formalize the dataset info file written to the repo ?
                info_messages[cfg['source']] = f"""
                    # {cfg['source']}
                    ## Version
                    Date of latest push: {datetime.date.today().isoformat()}
                    ## Splits
                    {cfg['source_split']}
                    ## Architecture and shape
                    {merged}
                    {merged.shape}
                    ## Comment
                    {cfg['comment']}
                """
                # end todo
        else:
            print(f"No data was loaded for dataset \"{cfg['source']}\".")

    if global_ds:
        global_merged = concatenate_datasets(global_ds)
        if args.push_to_hub:
            print("\nPushing to hub.")
            with tempfile.TemporaryDirectory() as tmpdir:
                repo = Repository(
                    local_dir=tmpdir, 
                    clone_from="LIMICS/PARTAGES" if args.make_commercial_version else "LIMICS/PARTAGES-research",
                    repo_type="dataset"
                )
                for key, msg in info_messages.items():
                    filename = f"{key}/{key}_info.md"
                    info_file = os.path.join(tmpdir, filename)
                    with open(info_file, 'w') as f:
                        f.write(msg)
                    repo.git_add(filename)
                parquet_file = os.path.join(tmpdir, "partages.parquet")
                global_merged.to_parquet(parquet_file)
                repo.git_add(parquet_file)
                repo.git_commit(commit_message=f"Adding new dataset version on {datetime.date.today().isoformat()}.")
                repo.git_push()
                print(f"Dataset pushed to {"LIMICS/PARTAGES" if args.make_commercial_version else "LIMICS/PARTAGES-research"}.")
        else:
            print("\nNot pushing to hub.")
    else:
        sys.tracebacklimit = 0 
        raise RuntimeError(f"No data was loaded.")

    ### TODO #3 -> cf the first todo
    df = pd.DataFrame(stats)
    totals = df[["nb_words", "nb_chars", "nb_docs"]].sum().rename("total")
    totals["mean_of_mean_words"] = df["mean_words"].mean()
    totals["std_of_mean_chars"] = df["std_chars"].std(ddof=0)
    totals["std_of_mean_words"] = df["std_words"].std(ddof=0)
    totals_df = totals.to_frame().T
    with pd.option_context('display.max_columns', None, 'display.width', 0):
        print(df)
        print()
        print(totals_df)
    # end todo

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF

if __name__ == "__main__":
    main()

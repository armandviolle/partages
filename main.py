import sys
import pandas as pd
import numpy as np
from datasets import concatenate_datasets
from loaders import REGISTRY
from huggingface_hub import HfFolder
import tempfile
import datetime
from loaders.utils import parse, load_config, get_nb_characters, get_nb_words


def main():

    args = parse()
    with open(args.hf_token, 'r') as f:
        hf_token = f.read()
    HfFolder.save_token(hf_token)
    all_cfg = load_config(args=args)

    stats = []
    all_ds = []

    for cfg in all_cfg:
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
                    stats.append(row)
                    all_ds.append(ds)
                except Exception as e:
                    print(f"Unavailable data split \"{split}\" for data_dir \"{subset}\".")
                    continue
        if not len(all_ds) > 0:
            sys.tracebacklimit = 0 
            raise RuntimeError(f"No data was loaded for dataset \"{cfg['source']}\".")

    merged = concatenate_datasets(all_ds)
    print(f"Shape of concatenated dataset: {merged.shape}")

    if args.push_to_hub:
        print("Pushing to hub\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            merged.save_to_disk(tmpdir)
            merged.push_to_hub(
                repo_id="LIMICS/PARTAGES",
                config_name=cfg["source"],
                commit_message=f"Pushed dataset {cfg['source']} on {datetime.date.today().isoformat()}"
            )
    else:
        print("Not pushing to hub\n")

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

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF

if __name__ == "__main__":
    main()

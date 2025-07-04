import os
import pandas as pd
import numpy as np
from datasets import concatenate_datasets
from loaders import REGISTRY
from huggingface_hub import HfFolder, HfApi, login, Repository
import tempfile
import datetime
from loaders.utils import parse, load_config, get_nb_characters, get_nb_words, generate_info_file, get_row_stats_individual


def main():

    args = parse()
    with open(args.hf_token, 'r') as f:
        hf_token = f.read()
    print(hf_token)
    login(token=hf_token)
    HfFolder.save_token(hf_token)
    all_cfg = load_config(args=args)

    stats = []
    global_ds = []
    commit_files = {}

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
                msg = generate_info_file(dataset=merged, source_name=cfg['source'], source_split=cfg['source_split'], comment=cfg['comment'])
                commit_files[cfg['source']] = [msg, concatenate_datasets(all_ds)] # cfg['target_split']]
        else:
            print(f"No data was loaded for dataset \"{cfg['source']}\".")
            raise ValueError()

    print(commit_files)

    df = pd.DataFrame(stats)
    totals = df[["nb_words", "nb_chars", "nb_docs"]].sum().rename("total")
    totals["mean_of_mean_words"] = df["mean_words"].mean()
    totals["std_of_mean_chars"] = df["std_chars"].std(ddof=0)
    totals["std_of_mean_words"] = df["std_words"].std(ddof=0)
    totals_df = totals.to_frame().T

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repository(
            local_dir=tmpdir, 
            clone_from="LIMICS/PARTAGES" if args.make_commercial_version else "LIMICS/PARTAGES-research",
            repo_type="dataset",
        )
        for key, val in commit_files.items():
            if not os.path.exists(os.path.join(tmpdir, key)):
                os.makedirs(os.path.join(tmpdir, key))
            # Writing and saving info file
            info_file_name = f"{key}/{key}_info.md"
            info_file = os.path.join(tmpdir, info_file_name)
            with open(info_file, 'w') as f:
                f.write(val[0])
            repo.git_add(info_file_name)
            # Saving dataframe 
            df_file_name = f"{key}/{key}.parquet"
            df_file = os.path.join(tmpdir, df_file_name)
            val[1].to_parquet(df_file)
            repo.git_add(df_file_name)
        # Writing stats_infos.md
        stats_infos_path = os.path.join(tmpdir, "stats_infos.md") # Ajout du fichier stats_infos.md à la racine
        with open(stats_infos_path, "w") as f:
            f.write(df.to_string(index=False, float_format="{:.2f}".format))
        repo.git_add("stats_infos.md")

        commit_msg = f"Updating dataset on {datetime.date.today().isoformat()}." if args.use_all_sources else f"Updating corpus {all_cfg[0]['source']} on {datetime.date.today().isoformat()}."
        repo.git_commit(commit_message=commit_msg)
        repo.git_push()

    with pd.option_context('display.max_columns', None, 'display.width', 0):
        print(df)
        print()
        print(totals_df)

    if args.use_all_sources:
        totals_df.to_csv("statistics.csv", index=False)

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF

if __name__ == "__main__":
    main()

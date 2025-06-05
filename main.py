import sys
import yaml
from argparse import ArgumentParser
from datasets import concatenate_datasets, disable_caching, load_dataset
from loaders import REGISTRY
from huggingface_hub import HfFolder, HfApi, login
import tempfile
import datetime

def parse():
    parser = ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="")
    return parser.parse_args()

def load_config(path="config/datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]


def main():

    args = parse()
    with open(args.hf_token, 'r') as f:
        hf_token = f.read()
    HfFolder.save_token(hf_token)

    # login()
    api = HfApi()
    # disable_caching()  # facultatif : si stream
    datasets_cfg = load_config()

    for cfg in datasets_cfg:
        print(type(cfg['subset']))
        print(f"Loading dataset {cfg["source"]}: using {REGISTRY[cfg["source"]]}")
        LoaderCls = REGISTRY[cfg["source"]]
    
        subsets = cfg['subset'] if isinstance(cfg['subset'], list) else [cfg['subset']]
        splits = cfg['source_split'] if isinstance(cfg['source_split'], list) else [cfg['source_split']]

        all_ds = []
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
                    print(f"Shape de {cfg['source']}: {ds.shape}")
                    print(f"{ds}\n")
                    all_ds.append(ds)
                except Exception as e:
                    print(f"Unavailable data split \"{split}\" for data_dir \"{subset}\".")
                    continue
        if not len(all_ds) > 0:
            sys.tracebacklimit = 0 
            raise RuntimeError(f"No data was loaded for dataset \"{cfg['source']}\".")
        merged = concatenate_datasets(all_ds)
        print(f"Shape on concatenated dataset: {merged.shape}")

        with tempfile.TemporaryDirectory() as tmpdir:
            merged.save_to_disk(tmpdir)
            merged.push_to_hub(
                repo_id="LIMICS/PARTAGES", 
                config_name=cfg["source"], 
                commit_message=f"Pushed dataset {cfg["source"]} on {datetime.date.today().isoformat()}"
            )

    # merged = concatenate_datasets(all_ds)
    # print(f"Shape on concatenated dataset: {merged.shape}")

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF

    # print(f"{len(merged):,} exemples après fusion et nettoyage")
    # merged.push_to_hub("dataset_name", token="hf_token")  # TODO : ajouter un token pour push_to_hub

if __name__ == "__main__":
    main()

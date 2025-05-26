import yaml
from datasets import concatenate_datasets, disable_caching, load_dataset
from loaders import REGISTRY


def load_config(path="config/datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]


def main():
    # disable_caching()  # facultatif : si stream
    datasets_cfg = load_config()
    all_ds = []
    for cfg in datasets_cfg:
        print(f"Loading dataset {cfg["source"]}: using {REGISTRY[cfg["source"]]}")
        LoaderCls = REGISTRY[cfg["source"]]
        loader = LoaderCls(
            source=cfg["source"], 
            path=cfg["path"], 
            subset=cfg["subset"], 
            source_split=cfg["source_split"]
        )
        ds = loader.load()
        print(f"Shape de {cfg["source"]}: {ds.shape}")
        print(f"{ds}\n")
        all_ds.append(ds)
    merged = concatenate_datasets(all_ds)
    print(f"Shape on concatenated dataset: {merged.shape}")

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF

    # print(f"{len(merged):,} exemples après fusion et nettoyage")
    # merged.push_to_hub("dataset_name", token="hf_token")  # TODO : ajouter un token pour push_to_hub

if __name__ == "__main__":
    main()

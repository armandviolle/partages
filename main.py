import yaml
from datasets import concatenate_datasets, disable_caching
from loaders import REGISTRY


def load_config(path="config/datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]

# TODO : ajouter colonne origine dataset
# TODO : définir structure finale dataset commun (text/label ? text/label/metadata?...)

def main():
    # disable_caching()  # facultatif : si stream

    datasets_cfg = load_config()

    all_ds = []
    for cfg in datasets_cfg:
        LoaderCls = REGISTRY[cfg["name"].capitalize()]
        loader = LoaderCls(cfg["hf_path"], cfg["split"], cfg["subset"])
        ds = loader.load()
        ds = ds.map(lambda x: {"dataset": cfg["name"]}) # ajout d'une colonne pour identifier le dataset d'origine
        print(f"""Shape de {cfg["name"]} = {ds.shape}""")
        all_ds.append(ds)

    merged = concatenate_datasets(all_ds)

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deplicate AF

    print(f"{len(merged):,} exemples après fusion et nettoyage")
    merged.push_to_hub("dataset_name", token="hf_token")  # TODO : ajouter un token pour push_to_hub

if __name__ == "__main__":
    main()

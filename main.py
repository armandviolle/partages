import sys
import yaml
from argparse import ArgumentParser
from datasets import concatenate_datasets, disable_caching, load_dataset
from loaders import REGISTRY
from huggingface_hub import HfFolder, HfApi, login
import tempfile
import datetime

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    else:
        return False

def parse():
    parser = ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--push_to_hub", type=str2bool, default=False)
    parser.add_argument("--use_all_sources", type=str2bool, default=True)
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--make_commercial_version", type=str2bool, default=True)
    return parser.parse_args()

def load_config(path="config/datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]


def main():

    args = parse()
    print(args)
    with open(args.hf_token, 'r') as f:
        hf_token = f.read()
    HfFolder.save_token(hf_token)
    api = HfApi()

    # disable_caching()  # facultatif : si stream
    all_cfg = load_config()

    print(args.use_all_sources, type(args.use_all_sources))

    print([cfg['source'] for cfg in all_cfg])

    if not args.use_all_sources:
        for cfg in all_cfg:        
            if args.source == cfg['source']:
                all_cfg = [cfg]
                print(all_cfg)
                break
        else: 
            sys.tracebacklimit = 0 
            raise RuntimeError(f"No available dataset named {args.source} in config.")
    
    print([cfg['source'] for cfg in all_cfg])

    if args.make_commercial_version:
        print("COMMERCIAL VERSION")
        print(f"Available datasets in config: {[cfg['source'] for cfg in all_cfg]}")
        tmp_cfg = []
        for cfg in all_cfg:
            if cfg['commercial_use']:
                tmp_cfg.append(cfg)
        all_cfg = tmp_cfg
        print(f"Remaning datasets after commercial use filtering: {[cfg['source'] for cfg in all_cfg]}")
    else:
        print("NON-COMMERCIAL VERSION")
        print(f"Available datasets in config: {[cfg['source'] for cfg in all_cfg]}")
    
    if len(all_cfg) < 1:
        sys.tracebacklimit = 0 
        raise RuntimeError(f"No available dataset(s) for given parametrization (check commercial use and source(s) given).")

    for cfg in all_cfg:
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

        if args.push_to_hub:
            print("Pushing to hub")
            with tempfile.TemporaryDirectory() as tmpdir:
                merged.save_to_disk(tmpdir)
                merged.push_to_hub(
                    repo_id="LIMICS/PARTAGES", 
                    config_name=cfg["source"], 
                    commit_message=f"Pushed dataset {cfg["source"]} on {datetime.date.today().isoformat()}"
                )
        else:
            print("Not pushing to hub")

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF

if __name__ == "__main__":
    main()

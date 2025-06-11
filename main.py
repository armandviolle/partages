import sys
from datasets import concatenate_datasets
from loaders import REGISTRY
from huggingface_hub import HfFolder
import tempfile
import datetime
from loaders.utils import parse, load_config


def main():

    args = parse()
    with open(args.hf_token, 'r') as f:
        hf_token = f.read()
    HfFolder.save_token(hf_token)
    all_cfg = load_config(args=args)

    for cfg in all_cfg:
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
                    print(f"Shape of {cfg['source']}-{subset}-{split}: {ds.shape}")
                    print(ds)
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
                    commit_message=f"Pushed dataset {cfg["source"]} on {datetime.date.today().isoformat()}"
                )
        else:
            print("Not pushing to hub\n")

    # déduplication simple
    # merged = deduplicate(merged, key_column="text") # TODO : deduplicate AF



if __name__ == "__main__":
    main()

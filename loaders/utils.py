import os, sys, yaml
from typing import Union
from pathlib import Path
from datasets import Dataset
from argparse import ArgumentParser


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


def read_config(path="config/datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]
    

def load_config(args):
    all_cfg = read_config()

    if not args.use_all_sources:
        for cfg in all_cfg:        
            if args.source == cfg['source']:
                all_cfg = [cfg]
                break
        else: 
            sys.tracebacklimit = 0 
            raise RuntimeError(f"No available dataset named {args.source} in config.")

    if args.make_commercial_version:
        print("\nCOMMERCIAL VERSION")
        print(f"Available datasets in config: {[cfg['source'] for cfg in all_cfg]}")
        tmp_cfg = []
        for cfg in all_cfg:
            if cfg['commercial_use']:
                tmp_cfg.append(cfg)
        all_cfg = tmp_cfg
        print(f"Remaining datasets after commercial use filtering: {[cfg['source'] for cfg in all_cfg]}\n")
    else:
        print("\nNON-COMMERCIAL VERSION")
        print(f"Available datasets in config: {[cfg['source'] for cfg in all_cfg]}\n")
    
    if len(all_cfg) < 1:
        sys.tracebacklimit = 0 
        raise RuntimeError(f"No available dataset(s) for given parametrization (check commercial use and source(s) given).")
    
    return all_cfg



def load_local(
    path: Union[str, Path], 
    split: Union[str, list], 
    data_dir: Union[str, list] = None, 
    streaming: bool = False, 
    trust_remote_code: bool = True
) -> Dataset:
    print(f"Loading from local path: {path} for split: {split}")
    all_texts = []
    if data_dir == "NACHOS": 
        if len(os.listdir(path)) == 1:
            with open(Path(path) / os.listdir(path)[0], 'r') as f:
                list_txt = f.read().splitlines()
            res = {"text": list_txt}
            return Dataset.from_dict(res)
        else:
            sys.tracebacklimit = 0 
            raise RuntimeError(f"None or Multiple data files available at {path}.")
    else:
        for root, dirs, files in os.walk(path):
            print(f"Searching for .txt files in {root}...")
            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            all_texts.append(f.read())
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
        if not all_texts:
            sys.tracebacklimit = 0
            raise RuntimeError(f"No .txt files found in {path} or its subdirectories.")
        else:
            return Dataset.from_dict({'text': all_texts}) 



def get_nb_characters(dataset: Dataset) -> int:
    """
    Returns the number of characters in the 'text' column of the dataset.
    """
    if 'text' not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'text' column.")

    return sum(len(text) for text in dataset['text'])

def get_nb_words(dataset: Dataset) -> int:
    """
    Returns the number of words in the 'text' column of the dataset.
    """
    if 'text' not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'text' column.")

    return sum(len(text.split()) for text in dataset['text'])

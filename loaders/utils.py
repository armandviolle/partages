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


def load_config(path="config/datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]


def load_local(
    path: Union[str, Path], 
    split: Union[str, list], 
    data_dir: Union[str, list] = None, 
    streaming: bool = False, 
    trust_remote_code: bool = True
) -> list:
    print(f"Loading from local path: {path} for split: {split}")
    all_texts = []
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
        # raw_dataset_items = [{"text": text_content} for text_content in all_texts]
        return Dataset.from_dict({'text': all_texts}) 

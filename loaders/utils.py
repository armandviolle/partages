import io, os, sys, yaml, gzip
import datetime
from typing import Union
from pathlib import Path
from datasets import Dataset, arrow_dataset
from argparse import ArgumentParser
from preprocessing.text_cleaning import cleaner


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
    

def read_compressed(path: Union[Path, str]) -> list:
    all_bytes = b""
    for part in sorted(os.listdir(path=path)):
        with open(path / part, 'rb') as f:
            all_bytes += f.read()
    with gzip.open(io.BytesIO(all_bytes), 'rt', encoding="utf-8") as res:
        return res.read().splitlines()


def generate_info_file(
    dataset: arrow_dataset.Dataset,
    source_name: str, 
    source_split: str, 
    comment: str,
    stats: dict
) -> str:
    return (f"# {source_name} \n"
            f"## Presentation \n"
            f"{comment} \n"
            f"## Version \n"
            f"Date of latest push: {datetime.date.today().isoformat()} \n"
            f"## Splits \n{source_split} \n"
            f"## Architecture and shape \n{dataset} \n"
            f"Shape: {dataset.shape}"
            f"## Stats \n" + "\n".join([f"{k}: {v}" for k, v in stats.items()]))

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
    if os.listdir(path=path)[0].endswith(".gz"):
        try: 
            all_texts = read_compressed(path=Path(path))
        except Exception as e:
            sys.tracebacklimit = 0 
            raise RuntimeError(f"Could not load data from {path}.")
        return Dataset.from_dict({'text': all_texts})
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



def clean_example(example, lower, rm_new_lines):
    example["text"] = cleaner(example["text"], do_lower=lower, rm_new_lines=rm_new_lines)
    return example


def get_row_stats_individual(source, stats):
    """
    Returns a dictionary with the statistics for each source.
    """
    for row in stats:
        if row['source'] == source:
            return {
                'nb_chars': row['nb_chars'],
                'nb_words': row['nb_words'],
                'nb_docs': row['nb_docs'],
                'mean_words': row['mean_words'],
                'std_chars': row['std_chars'],
                'std_words': row['std_words']
            }
        else:
            raise RuntimeError(f"Source {row['source']} is not available in statistics.")

import io
import os
import sys
import yaml
import gzip
import argparse
import datetime
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from datasets import Dataset, arrow_dataset
from preprocessing.text_cleaning import cleaner



def str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    else:
        return False



def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--push_to_hub", type=str2bool, default=False)
    parser.add_argument("--use_all_sources", type=str2bool, default=True)
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--make_commercial_version", type=str2bool, default=True)
    return parser.parse_args()



def read_config(path: Union[Path, str] = "config/datasets.yaml") -> dict:
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
    dataset:        arrow_dataset.Dataset,
    source_name:    str, 
    source_split:   str, 
    comment:        str,
    stats:          dict
) -> str:
    return (
        f"# {source_name} \n"
        f"## Presentation \n"
        f"{comment} \n"
        f"## Version \n"
        f"Date of latest push: {datetime.date.today().isoformat()} \n"
        f"## Splits \n{source_split} \n"
        f"## Architecture and shape \n{dataset} \n"
        f"Shape: {dataset.shape}"
        f"## Stats \n{pd.DataFrame(list(stats.values()), index=list(stats.keys())).to_string(index=False, float_format="{:.2f}".format)}\n"
    )



def load_config(args: argparse.Namespace) -> list:
    
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
    path:               Union[str, Path], 
    split:              Union[str, list], 
    data_dir:           Union[str, list] = None, 
    streaming:          bool = False, 
    trust_remote_code:  bool = True
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
        if all_texts:
            return Dataset.from_dict({'text': all_texts})
        else:
            sys.tracebacklimit = 0
            raise RuntimeError(f"No .txt or .parquet files found in {path} or its subdirectories.")



def compute_dataset_stats(
    dataset:        Dataset, 
    source_name:    str, 
    subset:         str, 
    split:          str,
) -> dict:
    if 'text' in dataset.column_names:
        words_lens = [len(txt.split()) for txt in dataset["text"]]
        chars_lens = [len(txt) for txt in dataset["text"]]
        row = {
            'nb_docs': dataset.shape[0],
            'nb_words': sum(words_lens),
            'mean_words': np.mean(words_lens),
            'std_words': np.std(words_lens, ddof=0),
            'nb_chars': sum(chars_lens),
            'mean_chars': np.mean(chars_lens),
            'std_chars': np.std(chars_lens, ddof=0),
        }
        return row
    else:
        sys.tracebacklimit = 0
        raise ValueError(f"Dataset \"{source_name}\" does not contain a 'text' column for subset \"{subset}\" and split \"{split}\".")



def pooled_variance(
    mode:   str, 
    row:    pd.Series, 
    avg:    float
) -> float:
    """
    Pooled variance for multiple subsets i: σ² = Σ(nᵢ(σᵢ² + (μᵢ - μ)²)) / (Σ nᵢ)
    """
    if mode == "word":
        return row['nb_docs'] * (row['std_words']**2 + (row['mean_words'] - avg)**2)
    elif mode == "char":
        return row['nb_docs'] * (row['std_chars']**2 + (row['mean_chars'] - avg)**2)
    else:
        sys.tracebacklimit = 0
        raise ValueError(f"Unkown mode option for pooled variance computation: {mode}.")



def weighted_avg_variance(
    mode:   str, 
    df:     pd.DataFrame, 
    total:  float,
) -> float:
    """
    Weighted average variance for multiple subsets i: σ² = (Σ nᵢσᵢ²) / (Σ nᵢ)
    """
    if mode == "word":
        return np.sqrt(np.sum([df["nb_docs"].iloc[i] * df["std_words"].iloc[i]**2 for i in range(len(df))]) / total)
    elif mode == "char":
        return np.sqrt(np.sum([df["nb_docs"].iloc[i] * df["std_chars"].iloc[i]**2 for i in range(len(df))]) / total)
    else:
        sys.tracebacklimit = 0
        raise ValueError(f"Unkown mode option for weighted average variance computation: {mode}.")



def update_row(
    base_row:   dict, 
    add_row:    dict,
) -> dict:
    new_row = { 
        'nb_docs':  base_row['nb_docs'] + add_row['nb_docs'], 
        'nb_words': base_row['nb_words'] + add_row['nb_words'], 
        'nb_chars': base_row['nb_chars'] + add_row['nb_chars'],
    }
    new_row['mean_words'] = new_row['nb_words'] / new_row['nb_docs']
    new_row['std_words'] = np.sqrt(np.sum([pooled_variance(mode="word", row=r, avg=new_row['mean_words']) for r in [base_row, add_row]]) / new_row['nb_docs'])
    new_row['mean_chars'] = new_row['nb_chars'] / new_row['nb_docs']
    new_row['std_chars'] = np.sqrt(np.sum([pooled_variance(mode="char", row=r, avg=new_row['mean_chars']) for r in [base_row, add_row]]) / new_row['nb_docs'])
    return new_row


        
def compute_global_stats(df: pd.DataFrame) -> None:
    totals = {
        'nb_docs': df['nb_docs'].sum(), 
        'nb_words': df['nb_words'].sum(), 
        'nb_chars': df['nb_chars'].sum(),
    }
    totals['mean_words'] = totals['nb_words'] / totals['nb_docs']
    totals['std_words'] = np.sqrt(np.sum([pooled_variance(mode="word", row=df.loc[source], avg=totals['mean_words']) for source in list(df.index)]) / totals['nb_docs'])
    # totals['std_words'] = weighted_avg_variance(mode="word", df=df, total=totals['nb_docs'])
    totals['mean_chars'] = totals['nb_chars'] / totals['nb_docs']
    totals['std_chars'] = np.sqrt(np.sum([pooled_variance(mode="char", row=df.loc[source], avg=totals['mean_chars']) for source in list(df.index)]) / totals['nb_docs'])
    # totals['std_chars'] = weighted_avg_variance(mode="char", df=df, total=totals['nb_docs'])
    df.loc["Total"] = pd.Series(totals)
        


def clean_example(
    example:        dict, 
    lower:          bool, 
    rm_new_lines:   bool,
) -> dict:
    example["text"] = cleaner(example["text"], do_lower=lower, rm_new_lines=rm_new_lines)
    return example

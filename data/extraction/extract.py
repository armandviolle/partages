from functools import wraps
from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets,
)
from config import data_register



PREPROCESS_REGISTRY = {}


# def prepare_dataset(path, split, name=None, data_files=None):
def prepare_dataset(cfg: dict, dataset_name: str):
    
    def decorator(fn):
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if isinstance(split, list):
                results = []
                for s in split:
                    data = load_dataset(path=cfg["path"], name=cfg["name"], data_files=cfg["data_files"], split=s)
                    # processed = fn(data, *args, **kwargs)
                    processed = fn(dataset_=data, split=split, dataset_name=dataset_name)
                    results.append(processed)
                return concatenate_datasets(results)
            else:
                data = load_dataset(path=cfg["path"], name=cfg["name"], data_files=cfg["data_files"], split=cfg["split"])
                return fn(dataset_=data, split=split, dataset_name=dataset_name)
        
        PREPROCESS_REGISTRY[fn.__name__] = wrapper
        return wrapper
        
    return decorator



# SIMSAMU
@prepare_dataset(cfg=data_register["SIMSAMU"], dataset_name="SIMSAMU")
def extract_texts(example):
    texts = [" ".join([t["text"] for t in mono["terms"]])
             for mono in example["monologues"]]
    return {"text": texts, "labels": [""] * len(texts)}


# WMT-16
@prepare_dataset(cfg=data_register["WMT16"], dataset_name="WMT16")
def extract_translation(example):
    return {
        "text": example["translation"]["en"],
        "labels": example["translation"]["fr"],
    }


# DEFT2021
@prepare_dataset(cfg=data_register["DEFT2021"], dataset_name="DEFT2021")
def preprocess_deft2021(dataset_, split, dataset_name):
    documents = []
    doc_ids = list(set(dataset_['document_id']))
    for id_ in doc_ids:
        rows = np.where(np.array(dataset_['document_id'])==id_)[0].tolist()
        documents.append({
            "id": id_, 
            "text": "\n".join([" ".join(dataset_['tokens'][i]) for i in rows]), 
            "name": name, 
            "split": split
        })
    return datasets.Dataset.from_list(documents)




def main():
    for name, func in PREPROCESS_REGISTRY.items():
        print(f"\nRunning {name}...")
        dataset = func()
        print(f"{name} returned {len(dataset)} samples")




if __name__ == "__main__":
    main()

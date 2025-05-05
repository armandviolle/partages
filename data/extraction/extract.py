from datasets import load_dataset, Dataset, concatenate_datasets



DATASETS = {
  "SIMSAMU": {
    "path": "medkit/simsamu",
    "type": "text",
    "split": "train", 
    "remove_columns": ["schemaVersion", "monologues"],
  },
  "WMT-16": {
    "path": "qanastek/WMT-16-PubMed", 
    "type": "translation",
    "files": "en-fr", 
    "split": "train",
    "remove_columns": [c for c in wmt.column_names if c != "translation"],
  }
}



def extract_texts(example):
  texts = [" ".join([t["text"] for t in mono["terms"]])
    for mono in example["monologues"]]
  return {"text": texts}

def extract_translation(example):
  return {
    "text": example["translation"]["en"],
    "labels": example["translation"]["fr"],
  }
  
# TODO

from datasets import load_dataset, Dataset, concatenate_datasets


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

from .base_loader import BaseLoader
from datasets import Dataset
from preprocessing.text_cleaning import cleaner

def extract_texts(example):
    texts = [" ".join([t["text"] for t in mono["terms"]]) for mono in example["monologues"]]
    return {"text": texts}

class Sisamu(BaseLoader):
    def postprocess(self, ds):
        sims_lists = ds.map(extract_texts, remove_columns=["schemaVersion", "monologues"])
        flat = {"text": [text for ex in sims_lists for text in ex["text"]]} # flatten in a single list
        flat["text"] = [cleaner(text) for text in flat["text"]] # PREPROCESSING text
        return Dataset.from_dict(flat)

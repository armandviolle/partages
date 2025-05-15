from .base_loader import BaseLoader
from preprocessing.text_cleaning import cleaner

def extract_translation(example):
    return {
        "text": cleaner(example["translation"]["fr"]), # With preprocessing
    }

class Wmt_16(BaseLoader):
    def postprocess(self, ds):
        return ds.map(
            extract_translation,
            remove_columns=[c for c in ds.column_names if c != "translation"]
        ).remove_columns(["translation"])

from .base_loader import BaseLoader
from preprocessing.text_cleaning import cleaner

def extract_translation(example):
    return {
        "text": cleaner(example["translation"]["fr"]), # With preprocessing
    }

class WMT16(BaseLoader):
    def postprocess(self, dataset, subset, split):
        res_ds = ds.map(
            extract_translation,
            remove_columns=[c for c in dataset.column_names if c != "translation"]
        ).remove_columns(["translation"])
        res_ds = res_ds.add_column("source", [self.source] * len(res_ds))
        res_ds = res_ds.add_column("subset", [subset] * len(res_ds))
        res_ds = res_ds.add_column("source_split", [split] * len(res_ds))
        return res_ds

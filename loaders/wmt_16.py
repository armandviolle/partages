from .base_loader import BaseLoader
from preprocessing.text_cleaning import cleaner

def extract_translation(example):
    return {
        "text": cleaner(example["translation"]["fr"]), # With preprocessing
    }

class WMT16(BaseLoader):
    def postprocess(self, ds, s):
        res_ds = ds.map(
            extract_translation,
            remove_columns=[c for c in ds.column_names if c != "translation"]
        ).remove_columns(["translation"])
        res_ds["dataset"] = [self.name] * len(res_ds)
        res_ds["split"] = [s] * len(res_ds)
        return res_ds

from .base_loader import BaseLoader
from datasets import Dataset


class WIKIPEDIA(BaseLoader):
    def postprocess(self, dataset: Dataset, subset: str, split: str):
        res = {
            "text": list(dataset['text']), 
            "source": [self.source] * len(dataset), 
            "subset": [subset] * len(dataset), 
            "source_split": [split] * len(dataset)
        }
        return Dataset.from_dict(res)

from .base_loader import BaseLoader
#from typing import override
from datasets import Dataset
from .utils import clean_example


class WIKIPEDIA(BaseLoader):

    def postprocess(self, dataset): 
        res = {
            "text": list(dataset['text']), 
            "source": [self.source] * len(dataset), 
            "subset": list(dataset['subset']), 
            "source_split": [self.split] * len(dataset)
        }
        return Dataset.from_dict(res)

    #@override
    def load(self):
        ds = Dataset.from_parquet(f"{self.path}/wikipedia.parquet")
        tmp_ds = self.postprocess(dataset=ds)
        return tmp_ds.map(clean_example, fn_kwargs={"lower": False, "rm_new_lines": False})

import os
from typing import override
from .base_loader import BaseLoader
from datasets import Dataset
from pathlib import Path


class NACHOS(BaseLoader):

    @override
    def load(self):
        subsets = self.subset if isinstance(self.subset, list) else [self.subset]
        for subset in subsets:
            data_path = Path(self.path) / Path(subset)
            if len(os.listdir(data_path)) == 1:
                with open(data_path / os.listdir(data_path)[0], 'r') as f:
                    list_txt = f.read().splitlines()
                res = {
                    "text": list_txt, 
                    "source": [self.source] * len(list_txt), 
                    "subset": [subset] * len(list_txt), 
                    "source_split": [self.split] * len(list_txt)
                }
                return Dataset.from_dict(res)
            else:
                sys.tracebacklimit = 0 
                raise RuntimeError(f"Multiple data files available at {self.source}/{subset}. Please select")

    def postprocess(self, dataset, subset, split):
        return dataset
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, concatenate_datasets
import sys

class BaseLoader(ABC):
    """Loader commun"""

    def __init__(self, name: str, path: str, data_dir: str = None, split: str = "train"):
        self.name       = name
        self.path       = path
        self.split      = split
        self.data_dir   = data_dir 
        self.stream     = False

    @abstractmethod
    def postprocess(self, ds: Dataset, d: str = None, s: str = "train") -> Dataset:
        """Spécifique au dataset : renommage de colonnes, filtrage, etc."""
        ...

    def load(self) -> Dataset:
        data_dirs = self.data_dir if isinstance(self.data_dir, list) else [self.data_dir]
        splits = self.split if isinstance(self.split, list) else [self.split]

        all_dirs = []
        for data_dir in data_dirs:
            all_splits = []
            for split in splits:
                try:
                    tmp_ds = load_dataset(path=self.path, data_dir=data_dir, split=split, streaming=self.stream, trust_remote_code=True)
                    tmp_ds = self.postprocess(ds=tmp_ds, d=data_dir, s=split)
                    all_splits.append(tmp_ds)
                except Exception as e:
                    print(f"Unavailable data split {split} for data_dir \"{data_dir}\".")
                    continue
            if len(all_splits) > 0:
                all_dirs += all_splits
            else:
                print(f"No data splits available for data_dir \"{data_dir}\" (probably unexistent datadir).")
        if len(all_dirs) > 0:
            return concatenate_datasets(all_dirs)
        else:
            sys.tracebacklimit = 0 
            raise RuntimeError(f"No data was loaded for dataset {self.name}.")
        return ds

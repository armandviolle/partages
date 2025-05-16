from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, concatenate_datasets

class BaseLoader(ABC):
    """Loader commun"""

    def __init__(self, name: str, path: str, data_dir: str = None, split: str = "train"):
        self.name       = name
        self.path       = path
        self.split      = split
        self.data_dir   = subset 
        self.stream     = False

    @abstractmethod
    def postprocess(self, ds: Dataset) -> Dataset:
        """Spécifique au dataset : renommage de colonnes, filtrage, etc."""
        ...

    def load(self) -> Dataset:
        if isinstance(self.split, list): 
            all_splits = []
            for s in self.split:
                tmp_ds = load_dataset(path=self.path, data_dir=self.data_dir, split=s, streaming=self.stream, trust_remote_code=True)
                tmp_ds = self.postprocess(ds=tmp_ds, s=s)
                all_splits.append(tmp_ds)
            ds = concatenate_datasets(all_splits)
        else:
            ds = load_dataset(path=self.path, data_dir=self.data_dir, split=self.split, streaming=self.stream, trust_remote_code=True)
            ds = self.postprocess(ds=ds, s=self.split)
        return ds

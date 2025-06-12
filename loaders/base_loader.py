import os, traceback
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from .utils import load_local, clean_example

class BaseLoader(ABC):
    """Common Loader"""

    def __init__(self, source: str, path: str, subset: str = None, source_split: str = "train"):
        self.source     = source
        self.path       = path
        self.split      = source_split
        self.subset     = subset 
        self.stream     = False

    @abstractmethod
    def postprocess(self, dataset: Dataset, subset: str = None, split: str = "train") -> Dataset:
        """Specific to each dataset: column renaming, filtering, etc."""
        ...

    def load(self) -> Dataset:
        load_fn = load_local if os.path.isdir(self.path) else load_dataset
        try:
            tmp_ds = load_fn(
                path=self.path, 
                data_dir=self.subset, 
                split=self.split, 
                streaming=self.stream, 
                trust_remote_code=True
            )
            ds = self.postprocess(dataset=tmp_ds, subset=self.subset, split=self.split)
            ds = ds.map(clean_example, fn_kwargs={"lower": False, "rm_new_lines": False})
            # print(ds[:5]["text"]) # DEBUG
            return ds
        except Exception as e:
            raise RuntimeError(f"Impossible to load this dataset:\n {traceback.format_exc()}")

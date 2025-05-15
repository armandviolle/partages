from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset

class BaseLoader(ABC):
    """Loader commun"""

    def __init__(self, hf_path: str, split: str, subset: str = None):
        self.hf_path = hf_path
        self.split   = split
        self.subset  = subset if subset else None
        self.stream = False

    @abstractmethod
    def postprocess(self, ds: Dataset) -> Dataset:
        """Spécifique au dataset : renommage de colonnes, filtrage, etc."""
        ...

    def load(self) -> Dataset:
        ds = load_dataset(path=self.hf_path, data_dir=self.subset, split=self.split, streaming=self.stream, trust_remote_code=True)
        ds = self.postprocess(ds)
        return ds

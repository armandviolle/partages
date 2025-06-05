from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, concatenate_datasets
import sys

class BaseLoader(ABC):
    """Loader commun"""

    def __init__(self, source: str, path: str, subset: str = None, source_split: str = "train"):
        self.source     = source
        self.path       = path
        self.split      = source_split
        self.subset     = subset 
        self.stream     = False

    @abstractmethod
    def postprocess(self, dataset: Dataset, subset: str = None, split: str = "train") -> Dataset:
        """Spécifique au dataset : renommage de colonnes, filtrage, etc."""
        ...

    def load(self) -> Dataset:
        # subsets = self.subset if isinstance(self.subset, list) else [self.subset]
        # splits = self.split if isinstance(self.split, list) else [self.split]

        # all_dirs = []
        # for subset in subsets:
        #     all_splits = []
        #     for split in splits:
        #         try:
        tmp_ds = load_dataset(
            path=self.path, 
            data_dir=self.subset, 
            split=self.split, 
            streaming=self.stream, 
            trust_remote_code=True
        )
        tmp_ds = self.postprocess(dataset=tmp_ds, subset=self.subset, split=self.split)
        #             all_splits.append(tmp_ds)
        #         except Exception as e:
        #             print(f"Unavailable data split \"{split}\" for data_dir \"{subset}\".")
        #             continue
        #     if len(all_splits) > 0:
        #         all_dirs += all_splits
        #     else:
        #         print(f"No data splits available for data_dir \"{subset}\" (probably unexistent data_dir).")
        # if len(all_dirs) > 0:
        #     return concatenate_datasets(all_dirs)
        # else:
        #     sys.tracebacklimit = 0 
        #     raise RuntimeError(f"No data was loaded for dataset \"{self.source}\".")
        return tmp_ds

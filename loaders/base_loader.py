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
        subsets = self.subset if isinstance(self.subset, list) else [self.subset]
        splits = self.split if isinstance(self.split, list) else [self.split]

        all_dirs = []
        for subset in subsets:
            all_splits = []
            for split in splits:
                rawData = None # To store data from either custom load or HF load
                try:
                    if hasattr(self, 'load_data') and callable(getattr(self, 'load_data')):
                        # Child loader has custom data loading
                        print(f"INFO: Attempting to use custom load_data method for {self.source}, subset {subset}, split {split}")
                        rawData = self.load_data(split=split)
                        if rawData is None: # Ensure load_data actually returned something
                             print(f"WARNING: Custom load_data for {self.source} returned None for split {split}. Skipping.")
                             continue
                    else:
                        # Default Hugging Face loading
                        print(f"INFO: Using Hugging Face load_dataset for {self.source}, path {self.path}, data_dir {subset}, split {split}")
                        rawData = load_dataset(
                            path=self.path,
                            data_dir=subset, # This is Hugging Face's concept of subset/config name
                            split=split,
                            streaming=self.stream,
                            trust_remote_code=True
                        )

                    tmp_ds = self.postprocess(dataset=rawData, subset=subset, split=split)
                    all_splits.append(tmp_ds)
                except Exception as e:
                    print(f"Error during data loading or postprocessing for {self.source}, subset '{subset}', split '{split}'. Path: '{self.path}'. Error: {e}")
                    # import traceback # Uncomment for detailed debugging
                    # print(traceback.format_exc()) # Uncomment for detailed debugging
                    print(f"Unavailable data split \"{split}\" for data_dir \"{subset}\" (or error in custom load_data).")
                    continue
            if len(all_splits) > 0:
                all_dirs += all_splits
            else:
                print(f"No data splits available for data_dir \"{subset}\" (probably unexistent data_dir).")
        if len(all_dirs) > 0:
            return concatenate_datasets(all_dirs)
        else:
            sys.tracebacklimit = 0 
            raise RuntimeError(f"No data was loaded for dataset \"{self.source}\".")
        return ds

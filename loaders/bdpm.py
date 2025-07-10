from typing import Dict, List, Optional
from datasets import Dataset
from .base_loader import BaseLoader


class BDPM(BaseLoader):
    """Loader for the BDPM dataset."""

    def postprocess(self, dataset: Dataset, subset: Optional[str] = None, split: str = "train") -> Dataset:
        """Format the raw dataset to a common schema.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess.
        subset : str, optional
            Name of the subset being processed. None by default.
        split : str
            Name of the split being processed. Defaults to "train".

        Returns
        -------
        Dataset
            The postprocessed dataset with "text", "source", "subset",
            and "source_split" columns.
        """
        res = {
            "text": list(dataset['text']), 
            "source": [self.source] * len(dataset), 
            "subset": [subset] * len(dataset), 
            "source_split": [split] * len(dataset)
        }
        return Dataset.from_dict(res)

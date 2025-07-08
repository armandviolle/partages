from typing import Dict, List, Optional
from datasets import Dataset
from .base_loader import BaseLoader

class MANTRA_GSC(BaseLoader):
    """Loader for the Mantra GSC dataset."""

    def postprocess(self, dataset: Dataset, subset: Optional[str], split: str) -> Dataset:
        """Format the raw dataset to a common schema.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess.
        subset : str, optional
            Name of the subset being processed.
        split : str
            Name of the split being processed.

        Returns
        -------
        Dataset
            The postprocessed dataset with "text", "source", "subset",
            and "source_split" columns.
        """
        res: Dict[str, List] = {
            "text": list(dataset["text"]),
            "source": [self.source] * len(dataset),
            "subset": [subset] * len(dataset),
            "source_split": [split] * len(dataset),
        }
        return Dataset.from_dict(res)

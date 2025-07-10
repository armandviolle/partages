from typing import Dict, List, Optional
from datasets import Dataset
from .base_loader import BaseLoader

class PXCORPUS(BaseLoader):
    """Loader for the PxCorpus dataset"""

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
        txt = dataset['text'][0].splitlines()
        res = {
            "text": list(txt),
            "source": [self.source] * len(dataset) * len(txt),
            "subset": [subset] * len(dataset) * len(txt),
            "source_split": [split] * len(dataset) * len(txt)
        }
        return Dataset.from_dict(res)
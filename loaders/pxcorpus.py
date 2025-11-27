import logging
from typing import Optional

from datasets import Dataset

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PXCORPUS(BaseLoader):
    """Loader for the PxCorpus dataset"""

    def postprocess(
        self, dataset: Dataset, data_dir: Optional[str] = None, split: str = "train"
    ) -> Dataset:
        """Format the raw dataset to a common schema.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess.
        data_dir : str, optional
            Name of the data_dir being processed. None by default.
        split : str
            Name of the split being processed. Defaults to "train".

        Returns
        -------
        Dataset
            The postprocessed dataset with "text", "source", "data_dir",
            and "source_split" columns.
        """
        txt = dataset["text"][0].splitlines()
        res = {
            "instruction": [None] * len(txt),
            "input": list(txt),
            "output": [None] * len(txt),
            "source": [self.source] * len(dataset) * len(txt),
            "data_dir": [data_dir] * len(dataset) * len(txt),
            "source_split": [split] * len(dataset) * len(txt),
        }
        return Dataset.from_dict(res)

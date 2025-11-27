import logging
from typing import Optional

from datasets import Dataset

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MQC(BaseLoader):
    """Loader for the MQC dataset"""

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
        document_sentences = [doc.splitlines() for doc in dataset["text"]]
        flattened = [" ".join(sentences) for sentences in document_sentences]
        n = len(flattened)

        res = {
            "instruction": [None] * n,
            "input": flattened,
            "output": [None] * n,
            "source": [self.source] * n,
            "data_dir": [data_dir] * n,
            "source_split": [split] * n,
        }

        return Dataset.from_dict(res)

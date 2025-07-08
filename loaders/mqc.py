from typing import Dict, List, Optional
from datasets import Dataset
from .base_loader import BaseLoader

class MQC(BaseLoader):
    """Loader for the MQC dataset"""

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
        document_sentences = [doc.splitlines() for doc in dataset["text"]]
        flattened = [" ".join(sentences) for sentences in document_sentences]
        n = len(flattened)

        res: Dict[str, List] = {
            "text": flattened,
            "source": [self.source] * n,
            "subset": [subset] * n,
            "source_split": [split] * n,
        }

        return Dataset.from_dict(res)
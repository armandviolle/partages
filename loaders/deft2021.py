import numpy as np
from typing import Dict, List, Any, Optional
from datasets import Dataset
from .base_loader import BaseLoader


class DEFT2021(BaseLoader):
    """Loader for the DEFT2021 dataset."""

    def postprocess(self, dataset: Dataset, subset: Optional[str] = None, split: str = "train") -> Dataset:
        """Format the DEFT2021 dataset to a common schema.

        This method groups tokens by document ID and concatenates them
        to form the document text.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess. It is expected to have
            'document_id' and 'tokens' columns.
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
        documents = []
        doc_ids = list(set(dataset["document_id"]))

        for id_ in doc_ids:
            rows = np.where(np.array(dataset["document_id"]) == id_)[0].tolist()
            documents.append(
                {
                    "text": "\n".join([" ".join(dataset["tokens"][i]) for i in rows]),
                    "source": self.source,
                    "subset": subset,
                    "source_split": split,
                }
            )
        new_dataset = Dataset.from_list(documents)
        return new_dataset

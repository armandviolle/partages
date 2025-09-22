import logging
from typing import Optional

from datasets import Dataset

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MANTRA_GSC(BaseLoader):
    """Loader for the Mantra GSC dataset."""

    def postprocess(
        self, dataset: Dataset, subset: Optional[str] = None, split: str = "train"
    ) -> Dataset:
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

        def gen():
            for row in dataset:
                yield {
                    "text": row["text"],  # type: ignore
                    "source": self.source,
                    "subset": subset,
                    "source_split": split,
                }

        return Dataset.from_generator(gen)  # type: ignore

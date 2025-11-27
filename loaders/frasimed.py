import logging
from typing import Optional

from datasets import Dataset

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class FRASIMED(BaseLoader):
    """Loader for the FRASIMED dataset."""

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

        def gen():
            for row in dataset:
                yield {
                    "instruction": None,
                    "input": row["text"],  # type: ignore
                    "output": None,
                    "source": self.source,
                    "data_dir": data_dir,
                    "source_split": split,
                }

        return Dataset.from_generator(gen)  # type: ignore

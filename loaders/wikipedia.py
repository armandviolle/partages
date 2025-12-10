import logging

from datasets import Dataset

from .base_loader import BaseLoader
from .utils import clean_example

logger = logging.getLogger(__name__)


class WIKIPEDIA(BaseLoader):
    """Loader for the Wikipedia dataset."""

    def postprocess(self, dataset: Dataset):
        """Format the raw dataset to a common schema.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess.
        data_dir : str, optional
            The data_dir of the dataset to process (e.g., "train", "validation", "test").
        split : str
            The split of the dataset to process (e.g., "train", "validation", "test"). Defaults to "train".

        Returns
        -------
        Dataset
            The postprocessed dataset with "text", "source", "data_dir" and "source_split" columns.
        """

        def gen():
            for row in dataset:
                yield {
                    "instruction": None,
                    "input": row["text"],  # type: ignore
                    "output": None,
                    "source": self.source,
                    "data_dir": row["subset"],  # type: ignore
                    "source_split": self.split,
                }

        return Dataset.from_generator(gen)  # type: ignore

    # @override
    def load(self):
        """
        Load the dataset from the specified path and apply postprocessing.

        Returns
        -------
        Dataset
            The loaded and postprocessed dataset.
        """
        ds = Dataset.from_parquet(f"{self.path}/wikipedia.parquet")
        tmp_ds = self.postprocess(dataset=ds)  # type: ignore
        return tmp_ds.map(  # type: ignore
            clean_example,
            fn_kwargs={"lower": False, "rm_new_lines": False, "columns": ["input"]},
        )

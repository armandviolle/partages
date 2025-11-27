import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from .utils import clean_example, load_local

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Common Loader.

    Base class for all data loaders. It provides a common interface for loading
    and processing datasets.

    Parameters
    ----------
    source : str
        Name of the data source.
    path : str
        Path to the dataset, can be a local directory or a Hugging Face Hub ID.
    data_dir : str, optional
        Name of the data_dir to load, if applicable. Defaults to None.
    source_split : str, optional
        Name of the split to load from the source. Defaults to "train".

    Attributes
    ----------
    source : str
        Name of the data source.
    path : str
        Path to the dataset.
    split : str
        Name of the split to load.
    data_dir : str or None
        Name of the data_dir to load.
    stream : bool
        Whether to stream the dataset. Defaults to False.
    """

    def __init__(
        self,
        source: str,
        path: str,
        adaptation_type: str,
        data_dir: Optional[str] = None,
        source_split: str = "train",
    ) -> None:
        self.source = source
        self.path = path
        self.split = source_split
        self.data_dir = data_dir
        self.adaptation_type = adaptation_type
        self.stream = False

    @abstractmethod
    def postprocess(
        self,
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
        data_dir: Optional[str] = None,
        split: str = "train",
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        """Perform dataset-specific postprocessing.

        This abstract method should be implemented by subclasses to handle tasks.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess.
        data_dir : str, optional
            Name of the data_dir being processed. Defaults to None.
        split : str, optional
            Name of the split being processed. Defaults to "train".

        Returns
        -------
        Dataset
            The processed dataset.
        """
        ...

    def load(self) -> Dataset:
        """Load the dataset.

        This method handles the loading of the dataset from either a local path
        or the Hugging Face Hub. It then applies postprocessing and a common
        preprocessing function.

        Returns
        -------
        Dataset
            The loaded and processed dataset.

        Raises
        ------
        RuntimeError
            If the dataset cannot be loaded.
        """
        load_fn = load_local if os.path.isdir(self.path) else load_dataset
        tmp_ds = load_fn(
            path=self.path,
            data_dir=self.data_dir,
            split=self.split,
            streaming=self.stream,
            trust_remote_code=True,
        )
        ds = self.postprocess(dataset=tmp_ds, data_dir=self.data_dir, split=self.split)  # type: ignore
        ds = ds.map(
            clean_example,
            fn_kwargs={
                "lower": False,
                "rm_new_lines": False,
                "columns": ["instruction", "input", "output"]
                if self.adaptation_type == "instruction-tuning"
                else ["input"],
            },
        )
        return ds  # type: ignore (dataset type ensuring by loading loop in main)

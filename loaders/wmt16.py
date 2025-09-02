import logging
from typing import Any, Dict, Optional

from datasets import Dataset
from preprocessing.text_cleaning import cleaner

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


def extract_translation(example: Dict[str, Any]) -> Dict[str, str]:
    """Extracts and cleans the French translation from a WMT16 example.

    Parameters
    ----------
    example : Dict[str, Any]
        An example from the WMT16 dataset, expected to have a "translation"
        key which is a dictionary containing a "fr" key for the French text.

    Returns
    -------
    Dict[str, str]
        A dictionary with a "text" key containing the cleaned French translation.
    """
    return {
        "text": cleaner(example["translation"]["fr"]),  # With preprocessing
    }


class WMT16(BaseLoader):
    """Loader for the WMT16 dataset."""

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
        res_ds = dataset.map(
            extract_translation,
            remove_columns=[c for c in dataset.column_names if c != "translation"],
        ).remove_columns(["translation"])
        res_ds = res_ds.add_column("source", [self.source] * len(res_ds))  # type: ignore
        res_ds = res_ds.add_column("subset", [subset] * len(res_ds))
        res_ds = res_ds.add_column("source_split", [split] * len(res_ds))
        return res_ds

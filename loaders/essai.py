import logging
from typing import Optional

from datasets import Dataset

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ESSAI(BaseLoader):
    """Loader for the ESSAI dataset"""

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
        # Parse dataset to extract only 3rd column and gather by sentence.
        all_texts = []

        file_content = dataset["text"][0]
        lines = file_content.splitlines()

        sentences = {}

        for line in lines:
            if line.strip():
                columns = line.split("\t")
                if len(columns) >= 6:
                    sentence_id = columns[0].strip()
                    word = columns[2].strip()

                    if word:
                        if sentence_id not in sentences:
                            sentences[sentence_id] = []
                        sentences[sentence_id].append(word)

        # Join words to form sentences
        for sentence_id, words in sentences.items():
            if words:
                sentence_text = " ".join(words)
                all_texts.append(sentence_text)

        res = {
            "text": all_texts,
            "source": [self.source] * len(all_texts),
            "subset": [subset] * len(all_texts),
            "source_split": [split] * len(all_texts),
        }
        return Dataset.from_dict(res)

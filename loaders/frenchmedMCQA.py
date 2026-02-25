import logging
from typing import List, Optional

from datasets import Dataset

from .base_loader import BaseLoader
from .utils import INSTRUCTIONS

logger = logging.getLogger(__name__)

map_id_qa = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def format_decoder(dataset: Dataset) -> List[str]:
    """Formats the FrenchMedMCQA dataset into a question-answer string format.

    Parameters
    ----------
    dataset : Dataset
        The input dataset, expected to have 'question', 'answer_a' through 'answer_e',
        and 'correct_answers' columns.

    Returns
    -------
    List[str]
        A list of strings, where each string is a formatted question-answer pair.
    """
    res = []
    for i in range(len(dataset)):
        tmp_str = f"### Question \n{dataset[i]['question']} \nA. {dataset[i]['answer_a']} \nB. {dataset[i]['answer_b']} \nC. {dataset[i]['answer_c']} \nD. {dataset[i]['answer_d']} \nE. {dataset[i]['answer_e']} \n### Réponse.s"
        for j in dataset[i]["correct_answers"]:
            tmp_str += (
                f"\n{map_id_qa[j]}. {dataset[i][f'answer_{map_id_qa[j].lower()}']}"
            )
        res.append(tmp_str)
    return res


def process_frenchmedmcqa(dataset: Dataset) -> Dataset:
    """
    Processes the FrenchMedMCQA dataset into a structured format with instructions,
    inputs, and outputs.

    Parameters
    ----------
    dataset : Dataset | IterableDataset
        The input dataset containing FrenchMedMCQA data.

    Returns
    -------
    Dataset | IterableDataset
        A list of dictionaries, each containing 'instruction', 'inputs', and 'outputs' keys
    """
    sample = {"instruction": [], "input": [], "output": []}
    for i in range(len(dataset)):
        sample["instruction"].append(
            "<instruction>\n%s\n</instruction>" % (INSTRUCTIONS["MCQall"])
        )
        sample["input"].append(
            "<input>\n\t<question>\n\t%s\n\t</question>\n\t<answers>\n\tA. %s\n\tB. %s\n\tC. %s\n\tD. %s\n\tE. %s\n\t</answers>\n</input>"
            % (
                dataset[i]["question"],
                dataset[i]["answer_a"],
                dataset[i]["answer_b"],
                dataset[i]["answer_c"],
                dataset[i]["answer_d"],
                dataset[i]["answer_e"],
            )
        )
        sample["output"].append(
            "<output>\n%s\n</output>"
            % (", ".join([map_id_qa[j] for j in dataset[i]["correct_answers"]]))
        )
    return Dataset.from_dict(sample)


class FRENCHMEDMCQA(BaseLoader):
    """Loader for the FrenchMedMCQA dataset."""

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
        tmp_ds = process_frenchmedmcqa(dataset=dataset)
        tmp_ds = tmp_ds.add_column(name="source", column=[self.source] * len(tmp_ds))  # type: ignore
        tmp_ds = tmp_ds.add_column("data_dir", [data_dir] * len(tmp_ds))
        tmp_ds = tmp_ds.add_column("source_split", [split] * len(tmp_ds))
        tmp_ds = tmp_ds.add_column("document_type", [self.document_type] * len(tmp_ds))
        return tmp_ds

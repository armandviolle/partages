from typing import List, Optional

from datasets import Dataset

from .base_loader import BaseLoader

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


class FRENCHMEDMCQA(BaseLoader):
    """Loader for the FrenchMedMCQA dataset."""

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
        tmp_ds = format_decoder(dataset=dataset)
        res = {
            "text": tmp_ds,
            "source": [self.source] * len(tmp_ds),
            "subset": [subset] * len(tmp_ds),
            "source_split": [split] * len(tmp_ds),
        }
        return Dataset.from_dict(res)

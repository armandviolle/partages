import logging

from datasets import Dataset

from .base_loader import BaseLoader

# from .utils import INSTRUCTIONS

logger = logging.getLogger(__name__)


def process_mediQAl(dataset: Dataset, data_dir: str) -> Dataset:
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
        instruction = (
            "Tu est un modèle expert en médecine et en sciences biomédicales, plus particulièrement en %s. Ta tâche consiste à répondre correctement à des QCMs médicaux comportant une seule bonne réponse. Lis attentivement la question et le contexte clinique quand il t'est donné pour t'aider à choisir la bonne réponse parmis les 5 proposées. Inidiquement uniquement la réponse correcte, sans justification."
            % (dataset[i]["medical_subject"])  # "medical_subject" is never null
            if data_dir == "mcqu"
            else "Tu est un modèle expert en médecine et en sciences biomédicales, plus particulièrement en %s. Ta tâche consiste à répondre correctement à des QCMs médicaux comportant plusieurs bonnes réponses (minimum 2). Lis attentivement la question et le contexte clinique quand il t'est donné pour t'aider à choisir les bonnes réponses parmis les 5 proposées. Inidiquement uniquement les réponses correctes, sans justification."
            % (dataset[i]["medical_subject"])  # "medical_subject" is never null
        )

        sample["instruction"].append(
            "<instruction>\n%s\n</instruction>" % (instruction)
        )
        sample["input"].append(
            "<input>\n\t%s<question>\n\t%s\n\t</question>\n\t<answers>\n\tA. %s\n\tB. %s\n\tC. %s\n\tD. %s\n\tE. %s\n\t</answers>\n</input>"
            % (
                "<clinical_case>\n\t%s\n\t</clinical_case>\n\t"
                % (dataset[i]["clinical_case"])
                if dataset[i]["clinical_case"] is not None
                else "",
                dataset[i]["question"],
                dataset[i]["answer_a"],
                dataset[i]["answer_b"],
                dataset[i]["answer_c"],
                dataset[i]["answer_d"],
                dataset[i]["answer_e"],
            )
        )
        sample["output"].append(
            "<output>\n%s\n</output>" % (dataset[i]["correct_answers"])
        )
    return Dataset.from_dict(sample)


class MEDIQAL(BaseLoader):
    """Loader for the FrenchMedMCQA dataset."""

    def postprocess(
        self, dataset: Dataset, data_dir: str, split: str = "train"
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
        tmp_ds = process_mediQAl(dataset=dataset, data_dir=data_dir)
        tmp_ds = tmp_ds.add_column(name="source", column=[self.source] * len(tmp_ds))  # type: ignore
        tmp_ds = tmp_ds.add_column("data_dir", [data_dir] * len(tmp_ds))
        tmp_ds = tmp_ds.add_column("source_split", [split] * len(tmp_ds))
        return tmp_ds

from dataclasses import dataclass, field
from typing import Any, Generator

import pandas as pd

import datasets
from datasets import (
    BuilderConfig,
    DatasetBuilder,
    DatasetInfo,
    Features,
    SplitGenerator,
    Value,
)


@dataclass
class PARTAGESConfig(BuilderConfig):
    """BuilderConfig for PARTAGES dataset.
    This class defines the configuration for loading and processing the PARTAGES dataset,
    including the sources to load, whether the dataset is for commercial use,
    and other parameters related to loading and processing the dataset.

    Parameters
    ----------
    sources : list[str]
        List of sources to load from the PARTAGES dataset.
    commercial_use : bool
        Whether the dataset is for commercial use. If True, loads the commercial use version of the dataset; otherwise, loads the research version.
    """

    sources: list[str] = field(default_factory=list)
    commercial_use: bool = True


class PARTAGES(DatasetBuilder):
    """DatasetBuilder for the PARTAGES dataset.
    This class defines how to load and process the PARTAGES dataset,
    including the features, splits, and example generation.

    Parameters
    ----------
    config : PARTAGESConfig
        Configuration for loading and processing the PARTAGES dataset.

    Returns
    -------
    datasets.Dataset
        A Hugging Face Dataset object containing the loaded and processed PARTAGES dataset.
    """

    BUILDER_CONFIG_CLASS = PARTAGESConfig
    config: PARTAGESConfig

    def _info(self) -> DatasetInfo:
        """Return the dataset metadata.
        This method defines the features of the dataset, which include
        "input", "instruction", "output", "source", "data_dir", "source_split", and "document_type".
        Each feature is of type string.

        Returns
        -------
        datasets.DatasetInfo
            The metadata of the dataset, including the defined features.
        """
        return datasets.DatasetInfo(
            features=Features(
                {
                    "input": Value("string"),
                    "instruction": Value("string"),
                    "output": Value("string"),
                    "source": Value("string"),
                    "data_dir": Value("string"),
                    "source_split": Value("string"),
                    "document_type": Value("string"),
                }
            )
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[SplitGenerator]:
        """Return the dataset splits.
        This method defines how to generate the dataset splits.
        It constructs the URLs for the dataset files based on the configuration (commercial use or research version)
        and uses the download manager to download the files.
        It then returns a list of SplitGenerator objects, which specify how to generate examples for each split.

        Parameters
        ----------
        dl_manager : datasets.DownloadManager
            The download manager to use for downloading the dataset files.

        Returns
        -------
        list[SplitGenerator]
            A list of SplitGenerator objects defining how to generate examples for each split.
        """
        if self.config.commercial_use:
            urls = [
                "hf://datasets/LIMICS/PARTAGES-sourced/%s/%s.parquet" % (src, src)
                for src in self.config.sources
            ]
        else:
            urls = [
                "hf://datasets/LIMICS/PARTAGES-Research-sourced/%s/%s.parquet"
                % (src, src)
                for src in self.config.sources
            ]

        downloaded_files = dl_manager.download(urls)

        return [
            SplitGenerator(
                name="train",
                gen_kwargs={"filepaths": downloaded_files},
            )
        ]

    def _generate_examples(
        self, filepaths: list[str]
    ) -> Generator[tuple[int, dict[str, Any | str]], Any, Any]:
        """Generate examples for the dataset.
        This method reads the dataset files (in parquet format) and yields examples in the form of a tuple of (index, example),
        where the example is a dictionary containing the fields "input", "instruction", "output", "source", "data_dir", "source_split", and "document_type".
        Each field is extracted from the corresponding columns in the parquet files, and if a field is missing, it defaults to an empty string.

        Parameters
        ----------
        filepaths : list[str]
            A list of file paths to the dataset files (in parquet format) to read and generate examples from.

        Yields
        ------
        tuple[int, dict]
            A tuple containing the index of the example and a dictionary with the fields
            "input", "instruction", "output", "source", "data_dir", "source_split", and "document_type".
        """
        idx = 0
        for filepath in filepaths:
            df = pd.read_parquet(filepath)
            for _, row in df.iterrows():
                yield (
                    idx,
                    {
                        field: row.get(field) or ""
                        for field in [
                            "input",
                            "instruction",
                            "output",
                            "source",
                            "data_dir",
                            "source_split",
                            "document_type",
                        ]
                    },
                )
                idx += 1

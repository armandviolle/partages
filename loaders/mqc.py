from datasets import Dataset
from .base_loader import BaseLoader

class MQC(BaseLoader):
    def postprocess(self, dataset, subset, split):
        """
        Processes the raw Hugging Face dataset OR locally loaded data for Mantra GSC French.
        If local: 'dataset' will be the list of dicts from load_data.
        If HF: 'dataset' is expected to be loaded via:
        datasets.load_dataset(path="bigbio/mantra_gsc", name="mantra_gsc_fr_source", split=split)
        """
        document_sentences = [doc.splitlines() for doc in dataset["text"]]
        flattened = [" ".join(sentences) for sentences in document_sentences]
        n = len(flattened)

        res = {
            "text": flattened,  # 1 chaîne par doc
            "source": [self.source] * n,
            "subset": [subset] * n,
            "source_split": [split] * n,
        }

        return Dataset.from_dict(res)
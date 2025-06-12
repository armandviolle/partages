from datasets import Dataset
from .base_loader import BaseLoader

class CAS(BaseLoader):
    def postprocess(self, dataset, subset, split):
        """
        Processes the raw Hugging Face dataset OR locally loaded data for Mantra GSC French.
        If local: 'dataset' will be the list of dicts from load_data.
        If HF: 'dataset' is expected to be loaded via:
        datasets.load_dataset(path="bigbio/mantra_gsc", name="mantra_gsc_fr_source", split=split)
        """
        res = {
            "text": list(dataset['text']),
            "source": [self.source] * len(dataset),
            "subset": [subset] * len(dataset),
            "source_split": [split] * len(dataset)
        }
        return Dataset.from_dict(res)

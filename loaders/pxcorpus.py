from datasets import Dataset
from .base_loader import BaseLoader

class PXCORPUS(BaseLoader):
    def postprocess(self, dataset, subset, split):
        """
        Processes the raw Hugging Face dataset OR locally loaded data for Mantra GSC French.
        If local: 'dataset' will be the list of dicts from load_data.
        If HF: 'dataset' is expected to be loaded via:
        datasets.load_dataset(path="bigbio/mantra_gsc", name="mantra_gsc_fr_source", split=split)
        """
        txt = dataset['text'][0].splitlines()
        res = {
            "text": list(txt),
            "source": [self.source] * len(dataset) * len(txt),
            "subset": [subset] * len(dataset) * len(txt),
            "source_split": [split] * len(dataset) * len(txt)
        }
        return Dataset.from_dict(res)
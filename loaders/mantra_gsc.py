from .base_loader import BaseLoader
from datasets import Dataset


class MANTRA_GSC(BaseLoader):

    def postprocess(self, dataset, subset, split):
        '''
        Processes the raw Hugging Face dataset for Mantra GSC French.
        The input 'dataset' is expected to be loaded via:
        datasets.load_dataset(path="bigbio/mantra_gsc", name="mantra_gsc_fr_source", split=split)
        '''

        processed_texts = []

        for item in dataset:
            if 'text' in item and isinstance(item['text'], str):
                processed_texts.append(item['text'])
            else:
                raise ValueError(f"Could not find text for item: {item}")
        res = {
            "text": processed_texts,
            "source": [self.source] * len(processed_texts),
            "subset": [subset] * len(processed_texts),
            "source_split": [split] * len(processed_texts)
        }

        return Dataset.from_dict(res)

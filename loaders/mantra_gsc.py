import os
from datasets import Dataset,concatenate_datasets
from .base_loader import BaseLoader # Assuming BaseLoader handles the overall loading flow

class MANTRA_GSC(BaseLoader):

    def load(self) -> Dataset:
        """
        Overlaods the load method to handle local data loading for the MANTRA GSC dataset.
        """
        subsets = self.subset if isinstance(self.subset, list) else [self.subset]
        splits = self.split if isinstance(self.split, list) else [self.split]

        all_dirs = []
        for subset in subsets:
            all_splits = []
            for split in splits:
                try:
                    print(f"INFO: Attempting to use custom load_data method for {self.source}, subset {self.subset}, split {split}")
                    rawData = self.load_local(split=split)
                    assert rawData is not None, f"WARNING: Custom load_data for {self.source} returned None for split {split}"
                    tmp_ds = self.postprocess(dataset=rawData, subset=subset, split=split)
                    all_splits.append(tmp_ds)
                except Exception as e:
                    print(f"Error during data loading or postprocessing for {self.source}, subset '{subset}', split '{split}'. Path: '{self.path}'. Error: {e}")
                    print(f"Unavailable data split \"{split}\" for data_dir \"{subset}\" (or error in custom load_data).")
                    continue
            if len(all_splits) > 0:
                all_dirs += all_splits
            else:
                print(f"No data splits available for data_dir \"{subset}\" (probably unexistent data_dir).")
        if len(all_dirs) > 0:
            return concatenate_datasets(all_dirs)
        else:
            raise RuntimeError(f"No data was loaded for dataset \"{self.source}\".")
        return ds


    def postprocess(self, dataset, subset, split):
        '''
        Processes the raw Hugging Face dataset OR locally loaded data for Mantra GSC French.
        If local: 'dataset' will be the list of dicts from load_data.
        If HF: 'dataset' is expected to be loaded via:
        datasets.load_dataset(path="bigbio/mantra_gsc", name="mantra_gsc_fr_source", split=split)
        '''
        processed_texts = []

        for item in dataset:
            if 'text' in item and isinstance(item['text'], str):
                processed_texts.append(item['text'])
            else:
                raise ValueError(f"Could not find 'text' string in item.")

        if not processed_texts and not (isinstance(dataset, list) and len(dataset) == 0 and os.path.isdir(self.path)):
            print(f"Warning: No texts were processed for subset '{subset}', split '{split}'. Check data source and loading logic.")

        res = {
            "text": processed_texts,
            "source": [self.source] * len(processed_texts), # self.source should be "MANTRA_GSC"
            "subset": [subset] * len(processed_texts),       # subset should be "French"
            "source_split": [split] * len(processed_texts)   # split should be "train"
        }

        return Dataset.from_dict(res)

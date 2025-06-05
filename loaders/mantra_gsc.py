import os
from datasets import Dataset
from .base_loader import BaseLoader # Assuming BaseLoader handles the overall loading flow

class MANTRA_GSC(BaseLoader):

    def load_data(self, split): # This method might be called by BaseLoader
        # 'split' argument might come from 'source_split' in datasets.yaml
        # self.path would be "datasets/mantra_gsc" from the config

        if os.path.isdir(self.path): # Check if path is a local directory
            print(f"Loading MANTRA_GSC from local path: {self.path} for split: {split}")
            all_texts = []
            for root, dirs, files in os.walk(self.path):
                print(f"Searching for .txt files in {root}...")
                for file_name in files:
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(root, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                all_texts.append(f.read())
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

            if not all_texts:
                print(f"No .txt files found in {self.path} or its subdirectories.")
                return []

            raw_dataset_items = [{"text": text_content} for text_content in all_texts]
            return raw_dataset_items

        else:
            print(f"Path {self.path} is not a local directory. MANTRA_GSC.load_data expects a local path.")
            raise FileNotFoundError(f"MANTRA_GSC.load_data was called, but {self.path} is not a local directory.")

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

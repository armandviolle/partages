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
            # Determine which subdirectories correspond to the "French" subset and "train" split.
            # For now, let's assume all .txt files in subdirs of self.path are relevant.
            # This might need refinement based on how "subset" and "split" map to local structure.

            # The user's files are in subdirectories like 'EMEA_ec22-cui-best_man'
            # We need to walk through these subdirectories.
            for root, dirs, files in os.walk(self.path):
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
                # Return an empty list of dicts, consistent with how postprocess expects data
                return []

            # Create a dataset-like structure that postprocess can handle
            # The current postprocess expects a list of dicts, each with a 'text' key
            raw_dataset_items = [{"text": text_content} for text_content in all_texts]
            return raw_dataset_items # This will be passed to postprocess

        else:
            # This else block implies that if self.path is not a local directory,
            # this loader expects the data to be loaded from Hugging Face by BaseLoader.
            # The BaseLoader would then call self.postprocess with the HF dataset.
            # So, this method should only be called if local loading is intended.
            # If BaseLoader is modified to call this, it should check os.path.isdir first.
            print(f"Path {self.path} is not a local directory. MANTRA_GSC.load_data expects a local path.")
            # Raise an error or return something that signals BaseLoader to use standard HF loading.
            # For now, raising an error makes it clear this path wasn't handled as expected for local.
            raise FileNotFoundError(f"MANTRA_GSC.load_data was called, but {self.path} is not a local directory.")


    def postprocess(self, dataset, subset, split):
        '''
        Processes the raw Hugging Face dataset OR locally loaded data for Mantra GSC French.
        If local: 'dataset' will be the list of dicts from load_data.
        If HF: 'dataset' is expected to be loaded via:
        datasets.load_dataset(path="bigbio/mantra_gsc", name="mantra_gsc_fr_source", split=split)
        '''
        processed_texts = []

        # Dataset can now be either a HuggingFace Dataset object or our list of dicts
        for item in dataset: # This should work for both HF dataset items and our dicts
            if 'text' in item and isinstance(item['text'], str):
                processed_texts.append(item['text'])
            else:
                # Provide more context for the error
                item_type = type(item)
                item_content_preview = str(item)[:100] # Preview of item content
                raise ValueError(f"Could not find 'text' string in item. Item type: {item_type}, Item preview: {item_content_preview}. Full item: {item}")

        if not processed_texts and not (isinstance(dataset, list) and len(dataset) == 0 and os.path.isdir(self.path)):
            # Avoid warning if it was an intentionally empty list from local loading (no files found)
            # but still warn if HF dataset is empty or other unexpected empty cases.
            print(f"Warning: No texts were processed for subset '{subset}', split '{split}'. Check data source and loading logic.")

        res = {
            "text": processed_texts,
            "source": [self.source] * len(processed_texts), # self.source should be "MANTRA_GSC"
            "subset": [subset] * len(processed_texts),       # subset should be "French"
            "source_split": [split] * len(processed_texts)   # split should be "train"
        }

        return Dataset.from_dict(res)

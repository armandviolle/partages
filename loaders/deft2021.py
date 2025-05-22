from .base_loader import BaseLoader
from preprocessing.text_cleaning import cleaner
import numpy as np
from datasets import Dataset


class DEFT2021(BaseLoader):
    def postprocess(self, dataset, subset, split):
        documents = []
        doc_ids = list(set(dataset['document_id']))
        for id_ in doc_ids:
            rows = np.where(np.array(dataset['document_id'])==id_)[0].tolist()
            documents.append({
                "text": "\n".join([" ".join(dataset['tokens'][i]) for i in rows]), 
                "source": self.source, 
                "subset": subset, 
                "source_split": split,
            })
        new_dataset = Dataset.from_list(documents)
        return new_dataset

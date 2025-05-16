from .base_loader import BaseLoader
from preprocessing.text_cleaning import cleaner
import numpy as np



class Deft2021(BaseLoader): # TODO : A MODIFIER
    def postprocess(self, ds, s):
        documents = []
        doc_ids = list(set(ds['document_id']))
        for id_ in doc_ids:
            rows = np.where(np.array(ds['document_id'])==id_)[0].tolist()
            documents.append({
                #"id": id_, 
                "text": "\n".join([" ".join(ds['tokens'][i]) for i in rows]), 
                "dataset": self.dataset_name, 
                "split": s
            })
        new_dataset = Dataset.from_list(documents)
        print(new_dataset)
        print(new_dataset[0])
        return new_dataset

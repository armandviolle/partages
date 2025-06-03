from .base_loader import BaseLoader
from datasets import Dataset
from preprocessing.text_cleaning import cleaner

map_id_qa = {
    0: "A", 
    1: "B", 
    2: "C", 
    3: "D", 
    4: "E"
}

def format_decoder(dataset):
    res = []
    for i in range(len(dataset)):
        tmp_str = f"### Question \n{dataset[i]['question']} \nA. {dataset[i]['answer_a']} \nB. {dataset[i]['answer_b']} \nC. {dataset[i]['answer_c']} \nD. {dataset[i]['answer_d']} \nE. {dataset[i]['answer_e']} \n### Réponse.s"
        for j in dataset[i]['correct_answers']:
            tmp_str += f"\n{map_id_qa[j]}. {dataset[i][f'answer_{map_id_qa[j].lower()}']}"
        res.append(tmp_str)
    return res



class FRENCHMEDMCQA(BaseLoader):

    def postprocess(self, dataset, subset, split):
        tmp_ds = format_decoder(dataset=dataset)
        res = {
            "text": tmp_ds, 
            "source": [self.source] * len(tmp_ds), 
            "subset": [subset] * len(tmp_ds),  
            "source_split": [split] * len(tmp_ds)
        }
        return Dataset.from_dict(res)
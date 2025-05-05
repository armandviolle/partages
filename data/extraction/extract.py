from datasets import load_dataset, Dataset, concatenate_datasets



DATASETS = {
  "SIMSAMU": {
    "path": "medkit/simsamu",
    "type": "text",
    "split": "train", 
    "remove_columns": ["schemaVersion", "monologues"],
  },
  "WMT-16": {
    "path": "qanastek/WMT-16-PubMed", 
    "type": "translation",
    "files": "en-fr", 
    "split": "train",
    "remove_columns": [c for c in wmt.column_names if c != "translation"],
  }, 
  # Format
  #   - 'id'
  #   - 'document_id'
  #   - 'tokens'
  #   - 'ner_tags'
  "DEFT-2021": {
    "path": "DrBenchmarck/DEFT2021", 
    "type": "annotation",
    "split": "train", 
    "read_text_fn": lambda x: [" ".join(x[i]['tokens']) for i in range(len(x))] # 
    "remove_columns": ['id', 'document_id']
}



def extract_texts(example):
  texts = [" ".join([t["text"] for t in mono["terms"]]) for mono in example["monologues"]]
  return {"text": texts}

def extract_translation(example):
  return {
    "text": example["translation"]["en"],
    "labels": example["translation"]["fr"],
  }
  
# TODO
# Essayer de généraliser l'extraction le plus possible en partant d'un dictionnaire
# -> Utiliser des lambda fonctions pour les étapes "simples" d'extraction e.g.:
#   convertir les fonctions ci-dessus en lambda fonctions initialisées dans le dictionnaire ??

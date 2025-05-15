"SIMSAMU": {
        "path": "medkit/simsamu", 
        "name": None,
        "data_files": None,
        "splits": ["train"],
        "text_column": "monologues",
        "additional_processing": True # cf fonction de Stéphane
    }, 
    "WMT-16": {
        "data_path": "qanastek/WMT-16-PubMed", 
        "name": "en-fr",
        "data_files": None,
        "splits": ["train"]
        "text_column": "translation",
        "additional_processing": True # extraction du texte des clés "fr" dans la dico de chaque ligne
    }, 
    "DEFT-2021": {
        "path": "DrBenchmarck/DEFT2021", 
        "name": None,
        "data_files": None,
        "splits": ["train", "validation", "test"], 
        "text_column": "tokens",
        "additional_processing": True # 1. On row-level, join lists on " " and 2. Concatenate rows with identical document_id, separated by "\n" or " ".
    }
    # Continue...
}

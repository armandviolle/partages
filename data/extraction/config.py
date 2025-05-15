data_register = {
    "SIMSAMU": {
        "path": "medkit/simsamu", 
        "name": None,
        "data_files": None,
        "split": "train",
        # "text_column": "monologues",
        "additional_processing": True
    }, 
    "WMT16": {
        "path": "qanastek/WMT-16-PubMed", 
        "name": "en-fr",
        "data_files": None,
        "split": "train",
        # "text_column": "translation",
        "additional_processing": True 
    }, 
    "DEFT2021": {
        "path": "DrBenchmark/DEFT2021", 
        "name": None,
        "data_files": None,
        "split": ["train", "validation", "test"], 
        # "text_column": "tokens",
        "additional_processing": True
    }
    # Continue...
}

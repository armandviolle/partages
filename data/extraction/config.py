DATASETS = {
    "SIMSAMU": {
        "data_path": "medkit/simsamu", 
        "data_file": "<data_file_name>",
        "splits": ["train"],
        "text_column": "<key_text_column>",
        "additional_processing": True
    }, 
    "WMT-16": {
        "data_path": "qanastek/WMT-16-PubMed", 
        "data_file": "<data_file_name>",
        "splits": ["train"]
        "text_column": "<key_text_column>",
        "additional_processing": True
    }, 
    "DEFT-2021": {
        "data_path": "DrBenchmarck/DEFT2021", 
        "data_file": "<data_file_name>",
        "splits": ["train"], 
        "text_column": "<key_text_column>",
        "additional_processing": True
    }
    # Continue
}

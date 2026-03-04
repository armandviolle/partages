# Data Deliverable - README

This document describes the first version of the {{commerciality}} PARTAGES corpus, as released in december 2025. 

## Overview

The availability of French biomedical data remains a major challenge for improving the multilingual capabilities of large language models (LLMs) in the medical domain.
We introduce and release the PARTAGES corpus, a collection of French biomedical texts compiled from a wide range of sources.

While similar datasets have been released in the past couple of years (NACHOS from DrBERT, JARGON), ours is the result of a greater scrutiny of the licensing terms of each source. Therefore, the PARTAGES corpus is fully compatible with research usage and is also distributed with a version compatible with commercial usage.
Here, we present the {{commerciality}} corpus released.


## Data Sources
The selected datasets for our corpus come from a variety of sources which can be categorized as follows: 

### Scientific
blabla
### Clinical
blabla

### Medicinal

## Preprocessing steps 

### Text cleaning
All documents were preprocessed using a the pipeline inspired by Le et al. (2020), including Unicode conversion and normalization, removal of characters outside standard French encoding, suppression of multiple spaces, and deletion of URLs

#TODO
- include additions by Aidan

### Deduplication
In order to avoid overfitting on redundant samples within our dataset, we applied an extra de-duplication step in our preprocessing. We used a very "classic" method based on minhash similarity. We used a similarity threshold of 0.85 and a number of allowed permutation f 128.

## Metadata

### Included Metadata
- **Schema information**: [Column names, data types, descriptions]
- **Data lineage**: [Information about data origin and transformations]
- **Quality metrics**: [Data quality indicators]
- **Timestamps**: [Creation/modification dates]

### Metadata Format
- [Specify how metadata is stored: separate file, embedded in Parquet, etc.]
- [Location of metadata files if separate]

## Data Schema
| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| column_1    | string    | [Description] | [Example value] |
| column_2    | integer   | [Description] | [Example value] |
| ...         | ...       | ...         | ...     |

## File Organization

PARTAGES/
├── fine-tuning/
│   ├── dataset1_part1.parquet
│   ├── dataset1_part2.parquet
│   └── ...
├── instruction-tuning/
│   ├── dataset2_part1.parquet
│   ├── dataset2_part2.parquet
│   └── ...
├── metadata/
│   ├── schema.json
│   └── data_dictionary.csv
└── README.md


PARTAGES-sourced/
├── source_1/
│   ├── dataset_part1.parquet
├── source_2
│   └── dataset_part2.parquet
├── ...
...
├── metadata
└── README.md


## Access and Usage
```python
from dataset import load_dataset

data = load_dataset(
	"LIMICS/PARTAGES", 
	split="train", 
	data_dir="fine-tuning"  # or "instruction-tuning"
	download_mode="force_redownload", 
	verification_mode="no_checks", 
)
```


## Contact
For questions or additional information, please contact:
- [Name/Team]
- [Email]
# PARTAGES

*Ajouter une description*.

## Fonctionnalités

- Chargement et filtrage des jeux de données selon l'usage commercial ou recherche.
- Agrégation de corpus depuis de multiples sources (fichiers locaux, archives, etc.).
- Nettoyage et prétraitement des textes.
- Calcul de statistiques globales et par corpus (nombre de documents, mots, caractères, moyennes, écarts-types).
- Génération automatique de fichiers d'information pour chaque dataset.
- Publication sur HuggingFace Hub (push automatique).

## Installation

### Clonage du repo

```sh
git clone https://github.com/armandviolle/partages.git
cd partages
```

### Création de l'environnement
Avec [miniconda](https://docs.conda.io/en/latest/miniconda.html) par exemple:
```sh
conda create -n <env_name> python=3.12
conda activate <env_name>
# vérifier que le cwd est bien partages/
pip install -r requirements.txt
```

## Développement

### Architecture du dépôt

```
partages/
│
├── main.py                # Script principal : exécution du pipeline de chargement, nettoyage, calcul stats et publication sur HugginFace
├── loaders/
│   └── utils.py           # Fonctions utilitaires pour le pipeline de chargment des donées.
│   └── base_loader.py     # Fichier implémentant la classe mère `BaseLoader` qui founrit une structure standard pour charger, nettoyer et préparer les corpus.
│   └── anses_rcp.py       # Fichier implémentant la classe fille héritant du `BaseLoader` en y aggrégeant les méthodes spécifiques au corpus concerné.wh
│   └── ...
├── datasets/              # Dossiers contenant les différents corpus médicaux 
│   ├── cas/
│   ├── frasimed/
│   ├── mantra_gsc/
│   └── ...
├── config/
│   └── datasets.yaml      # Configuration des sources et droits d’utilisation
│   └── hf-token.txt       # Token HuggingFace (non versionné, dans le .gitignore)
├── README.md              # Documentation du projet
└── requirements.txt       # Dépendances Python
```

- **main.py** : Orchestration du pipeline, gestion des arguments, publication sur HuggingFace.
- **loaders/utils.py** : Fonctions pour charger les données, filtrer selon les droits, nettoyer les textes, calculer les statistiques.
- **datasets/** : Dossiers contenant les différents corpus médicaux, organisés par source et format.
- **config/datasets.yaml** : Fichier de configuration listant les sources, leurs droits et paramètres.
- **config/hf-token.txt** : Fichier contenant le token hugging face de l'utilisateur.
- **README.md** : Fichier de présentation et de documentation du dépôt.
- **requirements.txt** : Liste des packages nécessaires et de leur versioning.

## Utilisation

### Produire un dataset à partir de tous les corpus disponibles:
- utilisation commerciale autorisée: `python main.py --hf_token config/hf-token.txt --use_all_sources True --make_commercial_version True --push_to_hub True`
- utilisation pour la recherche uniquement: `python main.py --hf_token config/hf-token.txt --use_all_sources True --make_commercial_version False --push_to_hub True`

### Ajout ou modification d'un dataset dans le dataset:
- utilisation commerciale autorisée: `python main.py --hf_token config/hf-token.txt --use_all_sources False --source <corpus_name> --make_commercial_version True --push_to_hub True`
- utilisation pour la recherche uniquement: `python main.py --hf_token config/hf-token.txt --use_all_sources False --source <corpus_name> --make_commercial_version True --push_to_hub True`

## Licence

Ce projet est distribué sous licence Apache 2.0.

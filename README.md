# partages

Produire un dataset à partir de tous les corpus disponibles:
- utilisation commerciale autorisée: `python main.py --hf_token /Users/armandviolle/Developer/partages/hf-token.txt --use_all_sources True --make_commercial_version True --push_to_hub True`
- utilisation pour la recherche uniquement: `python main.py --hf_token /Users/armandviolle/Developer/partages/hf-token.txt --use_all_sources True --make_commercial_version False --push_to_hub True`

Ajout ou modificaiton d'un dataset dans le dataset:
- utilisation commerciale autorisée: `python main.py --hf_token /Users/armandviolle/Developer/partages/hf-token.txt --use_all_sources False --source <corpus_name> --make_commercial_version True --push_to_hub True`
- utilisation pour la recherche uniquement: `python main.py --hf_token /Users/armandviolle/Developer/partages/hf-token.txt --use_all_sources False --source <corpus_name> --make_commercial_version True --push_to_hub True`


import os
import sys
import wikipediaapi
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser



ignored_med = [
    "Place des femmes en médecine", 
    "Clinicien", 
    "Personnalité de la médecine", 
    "Controverse en médecine", 
    "Distinction en médecine", 
    "Études de santé", 
    "Livre de santé et de bien-être", 
    "Médecine dans l'art et la culture", 
    "Recherche internationale sur le traumatisme cérébral de la naissance", 
]

ignored_bio = [
    "Biologie dans l'art et la culture", 
    "Biogéographie", 
    "Biologie des populations", 
    "Biologie en Chine", 
    "Biologie végétale", 
    "Biologiste", 
    "Biomathématiques", 
    "Biospéologie", 
    "Botanique", 
    "Classification scientifique des espèces", 
    "Discipline de la biologie", 
    "Distinction en biologie", 
    "Docteur en biologie", 
    "Écologie", 
    "Exobiologie", 
    "Géobiologie", 
    "Hydrobiologie", 
    "Littérature en biologie", 
    "Paléontologie", 
    "Philosophie de la biologie", 
    "Phrase biologique latine", 
    "Revue de biologie", 
    "Biologie et société", 
    "Zoologie", 
]
interrogation_bio = [
    "Biodiversité",
    "Biologie de l'évolution", 
    "Chronobiologie", 
    "Concept de biologie", 
    "Forme du raisonnement biologique", 
    "Nomenclature Biologique"
    "Ressource en biologie", 
    "Taxinomie", 
]

ignored_pharma = [
    "Membre de l'Académie nationale de pharmacie", 
    "Apothicairerie", 
    "Chaîne de pharmacie",  
    "Officine", 
    "Pharmacie ancienne de Florence", 
    "Pharmacie vétérinaire", 
    "Pharmacien", 
    "Pharmacienne", 
]
interrogation_pharma = [
    "Branche de la pharmacie", 
    "Association ou organisme lié à la pharmacie", 
    "Études de pharmacie",
    "Histoire de la pharmacie"
]

ignored_all = list(set(ignored_med + ignored_bio + ignored_pharma))



def get_filtered_categorymembers(categorymembers, titles, to_ignore=[], level=0, max_level=1):
    for k, c in categorymembers.items():
        if c.ns == wikipediaapi.Namespace.CATEGORY:
            if (level < max_level) and (k not in to_ignore):
                get_filtered_categorymembers(categorymembers=c.categorymembers, titles=titles, level=level+1, max_level=max_level)
        else: 
            titles.append(c.title)



def control_inter_duplication(titles : dict) -> None: 
    names = list(titles.keys())
    while names:
        name = names.pop(0)
        for t in titles[f"{name}"]:
            for cat in names:
                if t in titles[cat]:
                    titles[cat].remove(t)



def extract_wikipedia(
    category_names: str = ["Médecine", "Pharmacie", "Biologie"], 
    lang: str = "fr"
) -> list:
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=f"Articles liées au domaine biomédical", language=lang)
    titles = dict(zip(category_names, [[] for _ in range(len(category_names))]))
    texts = dict(zip(category_names, [[] for _ in range(len(category_names))]))
    for name in category_names:
        print(f"Extracting pages from category {name}.")
        tmp_titles = []
        try:
            category = wiki_wiki.page(f"Category:{name}")
            get_filtered_categorymembers(categorymembers=category.categorymembers, titles=tmp_titles, to_ignore=["Catégorie:" + name for name in ignored_all])
        except Exception as e:
            sys.tracebacklimit = 0
            print(f"No matching category found for {name}.")
        if tmp_titles:
            titles[name] = list(set(tmp_titles))
        else:
            print(f"WARNING: no pages found ")
    if any(list(titles.values())):
        control_inter_duplication(titles)
        tot = 0
        print()
        for key, val in titles.items():
            texts[key] = [wiki_wiki.page(f"{t}").text for t in val]
            print(f"{key}: {len(val)} pages.")
            tot += len(val)
        print(f"\nTotal: {tot} documents\n")
        return texts
    else:
        sys.tracebacklimit = 0
        raise ValueError("No matching category was found among the input one(s), no data loaded.")



def main():
    pages = extract_wikipedia()
    df = pd.DataFrame.from_dict({'subset': [], 'text': []})
    for key, val in pages.items():
        if val:
            df = pd.concat([df, pd.DataFrame.from_dict({'subset': [key]*len(val), 'text': val})])
    if df.empty:
        sys.tracebacklimit = 0
        raise ValueError("No data extracted for Wikipedia.")
    else: 
        df.to_parquet(f"datasets/WIKIPEDIA/wikipedia.parquet")



if __name__ == "__main__":
    main()
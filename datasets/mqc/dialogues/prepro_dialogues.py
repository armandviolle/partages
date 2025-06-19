"""
Les fichiers dialogues.txt contiennent des dialogues entre deux personnes et commencent par un identifiant de dialogue.
Ils sont organisés dans des dossiers nommés InterneX, où X est un numéro.
Ce script nettoie ces fichiers en supprimant l'identifiant de dialogue au début de chaque ligne.
"""

from pathlib import Path
import re

root_dir = Path(__file__).parent

# Pattern pour enlever "d12345 " ou "p67890 " en début de ligne
pattern = re.compile(r'^[dp]\d+\s*')

# On parcourt chaque sous-dossier InterneXX
for interne in root_dir.iterdir():
    if not interne.is_dir() or not interne.name.startswith('Interne'):
        continue

    # On cible tous les fichiers "dialogues*"
    for file_path in interne.glob('dialogues*'):
        if not file_path.is_file():
            continue

        stem, suffix = file_path.stem, file_path.suffix
        clean_path = interne / f"{stem}_clean{suffix}"

        with file_path.open('r', encoding='utf-8') as fin, \
             clean_path.open('w', encoding='utf-8') as fout:
            for line in fin:
                fout.write(pattern.sub('', line))

        file_path.unlink()
        print(f"→ créé {clean_path} et supprimé {file_path.name}")

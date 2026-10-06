# analyse_refus.py — taux de refus correct sur les questions hors données / hors sujet.
#
# Pour chaque version évaluée, affiche les 10 réponses concernées pour une vérification à la main,
# et calcule le taux de refus déclaré par l'agent (champ information_disponible, versions v1b et +).
# Lancement : uv run analyse_refus.py --version v1b
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Taux de refus correct d'une version évaluée")
parser.add_argument("--version", required=True, help="Dossier de résultats (ex. v0_baseline, v1a, v1b)")
args = parser.parse_args()

fichier = Path(__file__).resolve().parent / "eval" / "results" / args.version / "reponses.json"
reponses = json.loads(fichier.read_text(encoding="utf-8"))
cibles = [r for r in reponses if r["category"] in ("hors_donnees", "hors_sujet")]

print(f"=== {args.version} : {len(cibles)} questions hors données / hors sujet ===\n")
for r in cibles:
    dispo = r.get("information_disponible")   # absent (None) pour les versions sans agent
    print(f"[{r['id']}] {r['question']}")
    print(f"  information_disponible : {dispo}")
    print(f"  réponse : {r['answer'][:300].replace(chr(10), ' ')}")
    print()

declares = [r.get("information_disponible") for r in cibles]
if any(d is not None for d in declares):
    nb_refus = sum(d is False for d in declares)
    print(f"Taux de refus DÉCLARÉ par l'agent : {nb_refus} / {len(cibles)}")
    for cat in ("hors_donnees", "hors_sujet"):
        sous = [r for r in cibles if r["category"] == cat]
        print(f"  {cat:<13} {sum(r.get('information_disponible') is False for r in sous)} / {len(sous)}")
else:
    print("Pas de champ information_disponible (version sans agent) : compte les refus en lisant les réponses.")
print("\nÀ vérifier à la main : chaque réponse marquée False est-elle vraiment un refus, sans information inventée ?")
# scripts/voir_requetes_sql.py — affiche les requêtes SQL exécutées pour certaines questions d'une version.
# Lancement (depuis la racine) : uv run python -m scripts.voir_requetes_sql --version v2c_mixte --ids MX03 MX04 MX06 MX07
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Requêtes SQL exécutées par l'outil, question par question")
parser.add_argument("--version", required=True)
parser.add_argument("--ids", nargs="+", required=True)
args = parser.parse_args()

fichier = Path(__file__).resolve().parents[1] / "eval" / "results" / args.version / "reponses.json"
reponses = {r["id"]: r for r in json.loads(fichier.read_text(encoding="utf-8"))}

for id_ in args.ids:
    appels = reponses[id_].get("requetes_sql") or []
    print(f"\n--- {id_} : {len(appels)} appel(s) à l'outil SQL")
    print(f"  Réponse : {reponses[id_]['answer'][:150]}")
    for n, appel in enumerate(appels, 1):
        print(f"\n  [{n}] Question posée à l'outil : {appel.get('question')}")
        print(f"      SQL : {appel.get('requete')}")
        print(f"      Lignes : {appel.get('nb_lignes')} | Erreur : {appel.get('erreur')}")
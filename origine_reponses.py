# origine_reponses.py — d'où vient une réponse exacte : du contexte récupéré, ou des connaissances du modèle ?
#
# Pour chaque question choisie : affiche la réponse, puis vérifie si chaque élément attendu
# (eval/elements_attendus.json) figure dans la réponse ET dans le contexte récupéré.
#   - présent dans la réponse ET dans le contexte -> réponse fondée sur les données ;
#   - présent dans la réponse mais PAS dans le contexte -> le modèle l'a tiré de ses connaissances
#     (ou deviné) : réponse juste, mais non fondée, donc non fiable.
# Lancement : uv run origine_reponses.py --version v0_baseline_run1 --ids M02 M05 M08 M11
import argparse
import json
from pathlib import Path

from utils.exactitude import element_present

parser = argparse.ArgumentParser(description="Origine des éléments d'une réponse (contexte ou modèle)")
parser.add_argument("--version", required=True)
parser.add_argument("--ids", nargs="+", required=True)
args = parser.parse_args()

racine = Path(__file__).resolve().parent
reponses = json.loads((racine / "eval" / "results" / args.version / "reponses.json").read_text(encoding="utf-8"))
elements = json.loads((racine / "eval" / "elements_attendus.json").read_text(encoding="utf-8"))

for q in reponses:
    if q["id"] not in args.ids:
        continue
    contexte = "\n".join(q["contexts"])
    print(f"[{q['id']}] {q['question']}")
    print(f"  Réponse (début) : {q['answer'][:400].replace(chr(10), ' ')}")
    for e in elements.get(q["id"], []):
        dans_reponse = element_present(q["answer"], e)
        dans_contexte = element_present(contexte, e)
        if dans_reponse and dans_contexte:
            verdict = "fondé sur le contexte"
        elif dans_reponse:
            verdict = "ABSENT du contexte -> connaissances du modèle ou hasard"
        else:
            verdict = "absent de la réponse"
        print(f"    {str(e):<40} réponse: {'oui' if dans_reponse else 'non'} | contexte: "
              f"{'oui' if dans_contexte else 'non'} -> {verdict}")
    print()
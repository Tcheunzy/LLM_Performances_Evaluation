# voir_reponses.py — affiche des réponses précises d'une version évaluée, avec leur référence et leurs scores.
# Lancement : uv run voir_reponses.py --version v1b --ids S03 S05 S06 S07 S09 S10
import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Afficher des réponses d'une évaluation")
parser.add_argument("--version", required=True, help="Dossier de résultats (ex. v1b)")
parser.add_argument("--ids", nargs="+", required=True, help="Identifiants des questions (ex. S03 S05)")
args = parser.parse_args()

dossier = Path(__file__).resolve().parent / "eval" / "results" / args.version
reponses = json.loads((dossier / "reponses.json").read_text(encoding="utf-8"))
scores = pd.read_csv(dossier / "scores_par_question.csv").set_index("id")

for q in reponses:
    if q["id"] not in args.ids:
        continue
    print(f"[{q['id']}] {q['question']}")
    print(f"  Référence : {q['ground_truth']}")
    print(f"  Réponse   : {q['answer']}")
    if q["id"] in scores.index:
        s = scores.loc[q["id"]]
        print(f"  faithfulness = {s['faithfulness']:.2f} | answer_correctness = {s['answer_correctness']:.2f}")
    print()
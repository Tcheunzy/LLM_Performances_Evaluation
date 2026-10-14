# comparer_versions.py — comparer deux versions sur les MÊMES questions.
#
# Pourquoi : les moyennes de scores_par_categorie.csv ne sont pas comparables d'une version à l'autre quand
# certaines questions n'ont pas été notées (réponses en erreur), et la moyenne GLOBALE mélange des catégories
# de refus où RAGAS n'a pas de sens (answer_relevancy = 0 par conception).
#
# Le script :
#   1. garde uniquement les questions À RÉPONSE notées dans les DEUX versions, puis compare les moyennes ;
#   2. liste les questions à réponse dont answer_relevancy vaut 0, avec le début de la réponse (refus ?) ;
#   3. liste les plus gros écarts d'answer_correctness entre les deux versions, question par question.
# # Lancement (depuis la racine) : uv run python -m scripts.comparer_versions --versions v2 v2b
import argparse
import json
from pathlib import Path

import pandas as pd

METRIQUES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
CATEGORIES_A_REPONSE = ["simple", "bruitee", "multicriteres", "textuelle"]

parser = argparse.ArgumentParser(description="Comparer deux versions sur les mêmes questions")
parser.add_argument("--versions", nargs=2, required=True, metavar=("AVANT", "APRES"))
parser.add_argument("--ecarts", type=int, default=8, help="nombre de plus gros écarts à afficher")
parser.add_argument("--metrique", default="answer_correctness", choices=METRIQUES,
                    help="métrique dont on liste les écarts question par question (partie 3)")
parser.add_argument("--categorie", choices=CATEGORIES_A_REPONSE, help="limiter la partie 3 à une catégorie")
args = parser.parse_args()

RESULTATS = Path(__file__).resolve().parents[1] / "eval" / "results"


def charger(version: str) -> pd.DataFrame:
    """Scores RAGAS par question + id, catégorie, réponse, information_disponible et erreur (reponses.json)."""
    dossier = RESULTATS / version
    scores = pd.read_csv(dossier / "scores_par_question.csv")
    reponses = pd.DataFrame(json.loads((dossier / "reponses.json").read_text(encoding="utf-8")))
        # reindex : les anciennes versions (v0, v1a) n'ont pas toutes ces colonnes -> créées vides (NaN)
    infos = reponses.reindex(columns=["id", "question", "category", "answer", "information_disponible", "erreur"])

    if "id" in scores.columns:                        # cas simple : l'identifiant est déjà dans le CSV
        df = scores.drop(columns=[c for c in ["category", "question"] if c in scores.columns])
        df = df.merge(infos, on="id", how="left")
    else:                                             # sinon : rapprochement par le texte de la question
        col_q = "user_input" if "user_input" in scores.columns else "question"
        df = scores.merge(infos, left_on=col_q, right_on="question", how="left")
    if df["id"].isna().any():
        raise ValueError(f"{version} : {df['id'].isna().sum()} questions non rapprochées de reponses.json")

    df["erreur"] = df["erreur"].fillna(False).astype(bool)
    return df.set_index("id")[["category", "answer", "information_disponible", "erreur"] + METRIQUES]


avant_nom, apres_nom = args.versions
avant, apres = charger(avant_nom), charger(apres_nom)
pd.set_option("display.width", 200)

# --- 1. Moyennes sur les MÊMES questions
def notee(df):
    return df[(~df["erreur"]) & df["category"].isin(CATEGORIES_A_REPONSE) & df["answer_correctness"].notna()]

communes = notee(avant).index.intersection(notee(apres).index)
a_reponse = set(avant[avant["category"].isin(CATEGORIES_A_REPONSE)].index) | set(
    apres[apres["category"].isin(CATEGORIES_A_REPONSE)].index)
exclues = sorted(a_reponse - set(communes))
print(f"=== 1. Moyennes sur les {len(communes)} questions à réponse notées dans les deux versions ===")
print(f"Questions exclues (non notées ou en erreur dans au moins une version) : {exclues or 'aucune'}\n")

lignes = []
for cat in CATEGORIES_A_REPONSE + ["TOUTES"]:
    ids = communes if cat == "TOUTES" else [i for i in communes if avant.loc[i, "category"] == cat]
    if not len(ids):
        continue
    for m in METRIQUES:
        a, b = avant.loc[ids, m].mean(), apres.loc[ids, m].mean()
        lignes.append({"categorie": cat, "n": len(ids), "metrique": m,
                       avant_nom: round(a, 3), apres_nom: round(b, 3), "ecart": round(b - a, 3)})
tableau = pd.DataFrame(lignes)
print(tableau.pivot_table(index=["categorie", "n"], columns="metrique", values="ecart", sort=False)
      .reindex(columns=METRIQUES).to_string())
print(f"\n(écart = {apres_nom} − {avant_nom} ; détail complet ci-dessous)\n")
print(tableau.to_string(index=False))

# --- 2. answer_relevancy nulle sur des questions à réponse : refus ?
print(f"\n=== 2. Questions à réponse avec answer_relevancy = 0 ===")
for nom, df in [(avant_nom, avant), (apres_nom, apres)]:
    zeros = df[df["category"].isin(CATEGORIES_A_REPONSE) & (df["answer_relevancy"] == 0)]
    print(f"\n[{nom}] {len(zeros)} question(s)")
    for i, r in zeros.iterrows():
        print(f"  {i} ({r['category']}) information_disponible={r['information_disponible']} | "
              f"{str(r['answer'])[:140]}")

# --- 3. Plus gros écarts d'une métrique, question par question
m = args.metrique
ids = [i for i in communes if args.categorie is None or avant.loc[i, "category"] == args.categorie]
titre = f"{m}" + (f", catégorie {args.categorie}" if args.categorie else "")
print(f"\n=== 3. Les {args.ecarts} plus gros écarts de {titre} ({apres_nom} − {avant_nom}) ===")
ecarts = (apres.loc[ids, m] - avant.loc[ids, m]).dropna()
for i in ecarts.abs().sort_values(ascending=False).index[: args.ecarts]:
    print(f"\n{i} ({avant.loc[i, 'category']}) : {avant.loc[i, m]:.2f} -> {apres.loc[i, m]:.2f} ({ecarts[i]:+.2f})")
    print(f"  {avant_nom:>4} : {str(avant.loc[i, 'answer'])[:150]}")
    print(f"  {apres_nom:>4} : {str(apres.loc[i, 'answer'])[:150]}")
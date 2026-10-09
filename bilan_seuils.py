# bilan_seuils.py — compare les résultats d'une évaluation aux seuils cibles (grille à deux niveaux).
#
# Fonctionne sur les résultats DÉJÀ produits par evaluate_ragas.py (aucun appel à l'API) :
#   eval/results/<version>/scores_par_question.csv  et  eval/results/<version>/reponses.json
# Lancement : uv run bilan_seuils.py --version v1b
# Produit   : eval/results/<version>/bilan_seuils.csv
#
# Seuils fixés après l'analyse v0 -> v1b et FIGÉS avant l'évaluation de la v2.
import argparse
import json
from pathlib import Path

import pandas as pd

from utils.exactitude import score_exactitude

CATEGORIES_A_REPONSE = ["simple", "bruitee", "multicriteres", "textuelle"]
CATEGORIES_REFUS = ["hors_donnees", "hors_sujet"]

# Métriques bloquantes / de qualité : (seuil minimal = version bêta, seuil cible = production).
# Les métriques de diagnostic (context_recall, context_precision, answer_relevancy) n'ont pas de seuil.
SEUILS = {
    "faithfulness": {c: (0.80, 0.85) for c in CATEGORIES_A_REPONSE},
    "answer_correctness": {"simple": (0.60, 0.75), "bruitee": (0.60, 0.75),
                           "multicriteres": (0.60, 0.75), "textuelle": (0.50, 0.60)},
}
SEUIL_ACCEPTABLE = {"answer_correctness": 0.70, "faithfulness": 0.80}  # réponse « utilisable »

# Exactitude déterministe (utils/exactitude.py) : part des nombres / noms attendus présents dans la réponse.
# Ajoutée après la vérification manuelle de la v1b, et figée avec les autres seuils avant la v2.
SEUILS_EXACTITUDE = {"simple": (0.80, 0.90), "bruitee": (0.80, 0.90), "multicriteres": (0.60, 0.80)}


def statut(score: float, seuil_min: float, seuil_cible: float) -> str:
    if pd.isna(score):
        return "non mesuré"
    if score >= seuil_cible:
        return "cible"
    return "bêta" if score >= seuil_min else "non atteint"


def calculer_exactitude(reponses: pd.DataFrame, elements_attendus: dict) -> pd.DataFrame:
    """Score d'exactitude de chaque réponse qui a des éléments attendus (questions chiffrées / nominatives)."""
    lignes = []
    for _, q in reponses.iterrows():
        attendus = elements_attendus.get(q["id"])
        if attendus:
            lignes.append({"id": q["id"], "category": q["category"],
                           "exactitude": score_exactitude(q["answer"], attendus), "attendus": attendus})
    return pd.DataFrame(lignes)


def calculer_bilan(reponses: pd.DataFrame, scores: pd.DataFrame,
                   exactitude: pd.DataFrame | None = None) -> pd.DataFrame:
    """reponses : toutes les réponses (reponses.json) ; scores : scores RAGAS par question ;
    exactitude : scores d'exactitude par question (facultatif)."""
    par_categorie = scores.groupby("category")[list(SEUILS)].mean()
    lignes = []

    # 1. Seuils sur les moyennes par catégorie
    for metrique, par_cat in SEUILS.items():
        for categorie, (s_min, s_cible) in par_cat.items():
            if categorie in par_categorie.index:
                score = par_categorie.loc[categorie, metrique]
                lignes.append({"categorie": categorie, "metrique": metrique, "score": round(score, 3),
                               "seuil_min": s_min, "seuil_cible": s_cible,
                               "statut": statut(score, s_min, s_cible)})

    # 2. Taux de réponses acceptables (juste ET fidèle), question par question
    acceptable = ((scores["answer_correctness"] >= SEUIL_ACCEPTABLE["answer_correctness"])
                  & (scores["faithfulness"] >= SEUIL_ACCEPTABLE["faithfulness"]))
    for categorie in CATEGORIES_A_REPONSE:
        masque = scores["category"] == categorie
        if masque.any():
            lignes.append({"categorie": categorie, "metrique": "reponses_acceptables",
                           "score": f"{int(acceptable[masque].sum())}/{int(masque.sum())}",
                           "seuil_min": "—", "seuil_cible": "—", "statut": "indicateur"})

    # 2 bis. Exactitude déterministe (moyenne par catégorie + nombre de réponses exactes)
    if exactitude is not None and not exactitude.empty:
        for categorie, (s_min, s_cible) in SEUILS_EXACTITUDE.items():
            sous = exactitude[exactitude["category"] == categorie]
            if len(sous):
                score = sous["exactitude"].mean()
                lignes.append({"categorie": categorie, "metrique": "exactitude", "score": round(score, 3),
                               "seuil_min": s_min, "seuil_cible": s_cible,
                               "statut": statut(score, s_min, s_cible)})
                lignes.append({"categorie": categorie, "metrique": "reponses_exactes",
                               "score": f"{int((sous['exactitude'] == 1).sum())}/{len(sous)}",
                               "seuil_min": "—", "seuil_cible": "—", "statut": "indicateur"})

    # 3. Critères bloquants, sur TOUTES les réponses (y compris celles exclues de RAGAS)
    refus = reponses[reponses["category"].isin(CATEGORIES_REFUS)]
    if "information_disponible" in refus and refus["information_disponible"].notna().any():
        nb_refus = int((refus["information_disponible"] == False).sum())  # noqa: E712
        lignes.append({"categorie": "hors_donnees + hors_sujet", "metrique": "taux_de_refus (déclaré)",
                       "score": f"{nb_refus}/{len(refus)}", "seuil_min": "100 %", "seuil_cible": "100 %",
                       "statut": "cible" if nb_refus == len(refus) else "non atteint"})
    else:
        lignes.append({"categorie": "hors_donnees + hors_sujet", "metrique": "taux_de_refus",
                       "score": "à compter à la main (analyse_refus.py)", "seuil_min": "100 %",
                       "seuil_cible": "100 %", "statut": "non mesuré"})

    erreurs = reponses["erreur"] if "erreur" in reponses else pd.Series(False, index=reponses.index)
    nb_erreurs = int(erreurs.sum())
    lignes.append({"categorie": "toutes", "metrique": "reponses_en_erreur",
                   "score": f"{nb_erreurs}/{len(reponses)}", "seuil_min": "0 %", "seuil_cible": "0 %",
                   "statut": "cible" if nb_erreurs == 0 else "non atteint"})

    return pd.DataFrame(lignes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bilan d'une version évaluée par rapport aux seuils cibles")
    parser.add_argument("--version", required=True, help="Dossier de résultats (ex. v0_baseline_run1, v1a, v1b)")
    args = parser.parse_args()

    dossier = Path(__file__).resolve().parent / "eval" / "results" / args.version
    reponses = pd.DataFrame(json.loads((dossier / "reponses.json").read_text(encoding="utf-8")))
    scores = pd.read_csv(dossier / "scores_par_question.csv")

    fichier_elements = Path(__file__).resolve().parent / "eval" / "elements_attendus.json"
    exactitude = None
    if fichier_elements.exists():
        elements = json.loads(fichier_elements.read_text(encoding="utf-8"))
        exactitude = calculer_exactitude(reponses, elements)
        exactitude.to_csv(dossier / "exactitude_par_question.csv", index=False, encoding="utf-8-sig")
    else:
        print("eval/elements_attendus.json absent : lance d'abord build_elements_attendus.py\n")

    bilan = calculer_bilan(reponses, scores, exactitude)
    bilan.to_csv(dossier / "bilan_seuils.csv", index=False, encoding="utf-8-sig")

    print(f"=== Bilan {args.version} (min = bêta, cible = production) ===")
    print(bilan.to_string(index=False))
    print(f"\nEnregistré dans {dossier / 'bilan_seuils.csv'}")
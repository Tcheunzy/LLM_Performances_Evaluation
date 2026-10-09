# build_elements_attendus.py — éléments attendus (nombres, noms) pour la métrique d'exactitude.
#
# Produit eval/elements_attendus.json : {id de question: [éléments attendus]}.
# Le jeu de test (eval/testset.json) n'est PAS modifié : il reste figé. Les valeurs sont calculées
# avec pandas, exactement comme les références de build_testset.py.
# Lancement : uv run build_elements_attendus.py
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "inputs" / "regular NBA.xlsx"
TESTSET = ROOT / "eval" / "testset.json"
OUTPUT = ROOT / "eval" / "elements_attendus.json"

# Même chargement que build_testset.py
df = pd.read_excel(EXCEL, sheet_name="Données NBA", header=1)
df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
df = df.rename(columns={[c for c in df.columns if not isinstance(c, str)][0]: "3PM"})
equipes = pd.read_excel(EXCEL, sheet_name="Equipe")
NOM_EQUIPE = dict(zip(equipes.iloc[:, 0], equipes.iloc[:, 1]))


def joueur(nom: str) -> pd.Series:
    resultat = df[df["Player"] == nom]
    if resultat.empty:
        raise ValueError(f"Joueur introuvable : {nom!r}")
    return resultat.iloc[0]


def par_match(ligne: pd.Series, colonne: str) -> float:
    return float(round(ligne[colonne] / ligne["GP"], 1))


def noms(lignes: pd.DataFrame) -> list[str]:
    return list(lignes["Player"])


def nombre(x):
    """Convertit les types numpy en types Python (sérialisables en JSON)."""
    return int(x) if float(x).is_integer() else float(x)


h_booker, h_curry, h_gobert = joueur("Devin Booker"), joueur("Stephen Curry"), joueur("Rudy Gobert")
h_tatum, h_herro, h_harden, h_lavine = (joueur("Jayson Tatum"), joueur("Tyler Herro"),
                                        joueur("James Harden"), joueur("Zach LaVine"))
d = df.assign(ppg=(df["PTS"] / df["GP"]).round(1))
top_ast = df.nlargest(2, "AST")
jokic, sabonis = joueur("Nikola Jokić"), joueur("Domantas Sabonis")
top_blk = df.nlargest(1, "BLK").iloc[0]
top_pts = df.nlargest(1, "PTS").iloc[0]

elements = {
    # --- Questions simples : la valeur attendue
    "S01": [nombre(h_booker["3P%"])],
    "S02": [nombre(h_booker["PTS"])],
    "S03": [nombre(h_curry["OFFRTG"])],
    "S04": [[h_gobert["Team"], NOM_EQUIPE[h_gobert["Team"]]]],   # code OU nom complet
    "S05": [nombre(h_gobert["W"])],
    "S06": [nombre(h_tatum["L"])],
    "S07": [par_match(h_herro, "PTS")],
    "S08": [nombre(h_herro["REB"])],
    "S09": [nombre(h_harden["PIE"])],
    "S10": [par_match(h_harden, "3PA")],
    "S11": [nombre(h_lavine["FTA"])],
    "S12": [nombre(h_lavine["FTM"])],
    # --- Questions multicritères : les joueurs (et chiffres) qui doivent apparaître
    "M01": noms(df[df["3PA"] >= 300].nlargest(3, "3P%")),
    "M02": [top_pts["Player"], nombre(top_pts["PTS"])],
    "M03": noms(df.nlargest(3, "REB")),
    "M04": [top_ast.iloc[0]["Player"], nombre(top_ast.iloc[0]["AST"] - top_ast.iloc[1]["AST"])],
    "M05": noms(df[df["FTA"] >= 200].nlargest(1, "FT%")),
    "M06": noms(df[df["Age"] < 23].nlargest(1, "PTS")),
    "M07": noms(d[(d["GP"] >= 70) & (d["ppg"] >= 25)]),
    "M08": noms(df[df["GP"] >= 60].nlargest(1, "NETRTG")),
    "M09": [nombre(sabonis["REB"]), nombre(jokic["REB"]), par_match(sabonis, "REB"), par_match(jokic, "REB")],
    "M10": noms(df[(df["Team"] == "GSW") & (df["3PA"] >= 100)].nlargest(1, "3P%")),
    "M11": noms(df[df["Team"] == "OKC"].nlargest(2, "PTS")),
    "M12": [top_blk["Player"], [top_blk["Team"], NOM_EQUIPE[top_blk["Team"]]]],
}

# --- Questions bruitées : mêmes éléments que leur question d'origine (champ variante_de du jeu de test)
testset = json.loads(TESTSET.read_text(encoding="utf-8"))
for q in testset:
    if q.get("variante_de") in elements:
        elements[q["id"]] = elements[q["variante_de"]]

OUTPUT.write_text(json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(elements)} questions avec éléments attendus -> {OUTPUT}")
for id_, valeurs in elements.items():
    print(f"  {id_} : {valeurs}")
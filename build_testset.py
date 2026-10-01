# build_testset.py — construction du jeu de test d'évaluation (50 questions métier).
#
# Rôle : produire eval/testset.json, le « corrigé » utilisé par evaluate_ragas.py.
#   - Questions chiffrées (simples, multicritères, bruitées) : la réponse de référence est CALCULÉE
#     avec pandas directement depuis l'Excel → chaque référence est traçable jusqu'à sa formule.
#   - Questions textuelles (Reddit), hors données et hors sujet : la référence est ÉCRITE à la main.
#
# Ce fichier est figé une fois validé : le même jeu de test sert à évaluer toutes les versions (v0, v1, v2).
# Lancement : uv run build_testset.py
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "inputs" / "regular NBA.xlsx"
OUTPUT = ROOT / "eval" / "testset.json"

# ---------------------------------------------------------------------------
# 1. Chargement et nettoyage des données
# ---------------------------------------------------------------------------
# Les vrais en-têtes de « Données NBA » sont sur la 2e ligne du tableur (header=1).
df = pd.read_excel(EXCEL, sheet_name="Données NBA", header=1)

# Suppression des colonnes vides (« Unnamed ») en fin de tableau.
df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

# L'en-tête « 3PM » a été converti en heure (15:00:00) par Excel : on le renomme.
colonne_heure = [c for c in df.columns if not isinstance(c, str)]
df = df.rename(columns={colonne_heure[0]: "3PM"})

# Correspondance code équipe → nom complet (feuille « Equipe »).
equipes = pd.read_excel(EXCEL, sheet_name="Equipe")
NOM_EQUIPE = dict(zip(equipes.iloc[:, 0], equipes.iloc[:, 1]))

# ATTENTION — vérifié sur les données : contrairement au dictionnaire des données,
# PTS, REB, AST, BLK, 3PA, 3PM, FTA, FTM... sont des TOTAUX sur la saison,
# alors que Min est une moyenne par match. Les moyennes par match sont donc calculées (total / GP).


# ---------------------------------------------------------------------------
# 2. Fonctions utilitaires
# ---------------------------------------------------------------------------
def joueur(nom: str) -> pd.Series:
    """Renvoie la ligne du joueur choisi (erreur explicite si le nom est introuvable)."""
    resultat = df[df["Player"] == nom]
    if resultat.empty:
        raise ValueError(f"Joueur introuvable : {nom!r}")
    return resultat.iloc[0]


def par_match(ligne: pd.Series, colonne: str) -> float:
    """Moyenne par match d'une statistique cumulée sur la saison."""
    return round(ligne[colonne] / ligne["GP"], 1)


def fr(x) -> str:
    """Format français des nombres décimaux (44.6 → 44,6)."""
    return str(x).replace(".", ",")


def liste(lignes: pd.DataFrame, colonne: str, unite: str = "") -> str:
    """Formate un classement : 'Joueur A (valeur), Joueur B (valeur)...'."""
    return ", ".join(f"{r['Player']} ({fr(r[colonne])}{unite})" for _, r in lignes.iterrows())


testset: list[dict] = []
references: dict[str, str] = {}  # id → référence, pour que les questions bruitées réutilisent celle de leur question propre


def ajouter(id_: str, question: str, ground_truth: str, category: str, source: str, variante_de: str | None = None):
    """Ajoute une question au jeu de test."""
    item = {
        "id": id_,
        "question": question,
        "ground_truth": ground_truth,
        "category": category,
        "source_attendue": source,
    }
    if variante_de:
        item["variante_de"] = variante_de
    testset.append(item)
    references[id_] = ground_truth


# ---------------------------------------------------------------------------
# 3. Questions SIMPLES (12) — une valeur pour un joueur
# ---------------------------------------------------------------------------
h = joueur("Devin Booker")
ajouter("S01", "Quel est le pourcentage de réussite à 3 points (3P%) de Devin Booker ?",
        f"Le 3P% de Devin Booker est de {fr(h['3P%'])} %.", "simple", "excel")
ajouter("S02", "Combien de points Devin Booker a-t-il marqués au total sur la saison ?",
        f"Devin Booker a marqué {h['PTS']} points au total sur la saison.", "simple", "excel")

h = joueur("Stephen Curry")
ajouter("S03", "Quel est le rating offensif (OFFRTG) de Stephen Curry ?",
        f"Le rating offensif (OFFRTG) de Stephen Curry est de {fr(h['OFFRTG'])}.", "simple", "excel")

h = joueur("Rudy Gobert")
ajouter("S04", "Pour quelle équipe joue Rudy Gobert ?",
        f"Rudy Gobert joue pour {h['Team']} ({NOM_EQUIPE[h['Team']]}).", "simple", "excel")
ajouter("S05", "Combien de matchs l'équipe de Rudy Gobert a-t-elle gagnés lorsqu'il était sur le terrain ?",
        f"L'équipe de Rudy Gobert a gagné {h['W']} des {h['GP']} matchs qu'il a joués.", "simple", "excel")

h = joueur("Jayson Tatum")
ajouter("S06", "Combien de matchs l'équipe de Jayson Tatum a-t-elle perdus lorsqu'il était sur le terrain ?",
        f"L'équipe de Jayson Tatum a perdu {h['L']} des {h['GP']} matchs qu'il a joués.", "simple", "excel")

h = joueur("Tyler Herro")
ajouter("S07", "Combien de points Tyler Herro marque-t-il en moyenne par match ?",
        f"Tyler Herro marque en moyenne {fr(par_match(h, 'PTS'))} points par match "
        f"({h['PTS']} points en {h['GP']} matchs).", "simple", "excel")
ajouter("S08", "Quel est le nombre total de rebonds de Tyler Herro sur la saison ?",
        f"Tyler Herro a capté {h['REB']} rebonds au total sur la saison.", "simple", "excel")

h = joueur("James Harden")
ajouter("S09", "Quel est le PIE (Player Impact Estimate) de James Harden ?",
        f"Le PIE de James Harden est de {fr(h['PIE'])}.", "simple", "excel")
ajouter("S10", "Combien de tirs à 3 points James Harden tente-t-il en moyenne par match ?",
        f"James Harden tente en moyenne {fr(par_match(h, '3PA'))} tirs à 3 points par match "
        f"({h['3PA']} tentatives en {h['GP']} matchs).", "simple", "excel")

h = joueur("Zach LaVine")
ajouter("S11", "Quel est le nombre de lancers francs tentés par Zach LaVine sur la saison ?",
        f"Zach LaVine a tenté {h['FTA']} lancers francs sur la saison.", "simple", "excel")
ajouter("S12", "Quel est le nombre de lancers francs réussis par Zach LaVine sur la saison ?",
        f"Zach LaVine a réussi {h['FTM']} lancers francs sur la saison.", "simple", "excel")

# ---------------------------------------------------------------------------
# 4. Questions MULTICRITÈRES (12) — filtres, classements, comparaisons, agrégations
# ---------------------------------------------------------------------------
top = df[df["3PA"] >= 300].nlargest(3, "3P%")
ajouter("M01", "Quels sont les 3 meilleurs pourcentages à 3 points parmi les joueurs ayant tenté au moins 300 tirs à 3 points ?",
        f"Les 3 meilleurs 3P% (au moins 300 tentatives) : {liste(top, '3P%', ' %')}.", "multicriteres", "excel")

top = df.nlargest(1, "PTS").iloc[0]
ajouter("M02", "Quel joueur a marqué le plus de points sur la saison, et combien ?",
        f"{top['Player']} ({top['Team']}) a marqué le plus de points : {top['PTS']} au total.", "multicriteres", "excel")

top = df.nlargest(3, "REB")
ajouter("M03", "Quels sont les 3 meilleurs rebondeurs de la saison en nombre total de rebonds ?",
        f"Les 3 meilleurs rebondeurs (total) : {liste(top, 'REB')}.", "multicriteres", "excel")

top = df.nlargest(2, "AST")
a, b = top.iloc[0], top.iloc[1]
ajouter("M04", "Quel joueur a délivré le plus de passes décisives sur la saison, et combien de plus que le deuxième ?",
        f"{a['Player']} a délivré le plus de passes décisives ({a['AST']}), soit {a['AST'] - b['AST']} de plus "
        f"que le deuxième, {b['Player']} ({b['AST']}).", "multicriteres", "excel")

top = df[df["FTA"] >= 200].nlargest(1, "FT%").iloc[0]
ajouter("M05", "Quel joueur a le meilleur pourcentage aux lancers francs parmi ceux ayant tenté au moins 200 lancers francs ?",
        f"{top['Player']} a le meilleur pourcentage aux lancers francs (au moins 200 tentatives) : "
        f"{fr(top['FT%'])} % sur {top['FTA']} tentatives.", "multicriteres", "excel")

top = df[df["Age"] < 23].nlargest(1, "PTS").iloc[0]
ajouter("M06", "Parmi les joueurs de moins de 23 ans, lequel a marqué le plus de points sur la saison ?",
        f"{top['Player']} ({top['Age']} ans, {top['Team']}) a marqué le plus de points parmi les moins de 23 ans : "
        f"{top['PTS']} points.", "multicriteres", "excel")

d = df.assign(ppg=(df["PTS"] / df["GP"]).round(1))
sel = d[(d["GP"] >= 70) & (d["ppg"] >= 25)].sort_values("ppg", ascending=False)
ajouter("M07", "Combien de joueurs ont joué au moins 70 matchs avec une moyenne d'au moins 25 points par match ? Lesquels ?",
        f"{len(sel)} joueurs : {liste(sel, 'ppg', ' pts/match')}.", "multicriteres", "excel")

top = df[df["GP"] >= 60].nlargest(1, "NETRTG").iloc[0]
ajouter("M08", "Quel joueur ayant disputé au moins 60 matchs a le meilleur Net Rating (NETRTG) ?",
        f"{top['Player']} ({top['Team']}) a le meilleur Net Rating parmi les joueurs à 60 matchs ou plus : "
        f"{fr(top['NETRTG'])}.", "multicriteres", "excel")

j, s = joueur("Nikola Jokić"), joueur("Domantas Sabonis")
ajouter("M09", "Compare le nombre de rebonds de Nikola Jokić et de Domantas Sabonis sur la saison, au total et par match.",
        f"Domantas Sabonis : {s['REB']} rebonds ({fr(par_match(s, 'REB'))} par match) ; "
        f"Nikola Jokić : {j['REB']} rebonds ({fr(par_match(j, 'REB'))} par match). "
        f"Sabonis en a capté {s['REB'] - j['REB']} de plus.", "multicriteres", "excel")

top = df[(df["Team"] == "GSW") & (df["3PA"] >= 100)].nlargest(1, "3P%").iloc[0]
ajouter("M10", "Chez les Golden State Warriors, quel joueur a le meilleur pourcentage à 3 points parmi ceux ayant tenté au moins 100 tirs à 3 points ?",
        f"{top['Player']} a le meilleur 3P% des Warriors (au moins 100 tentatives) : "
        f"{fr(top['3P%'])} % sur {top['3PA']} tentatives.", "multicriteres", "excel")

top = df[df["Team"] == "OKC"].nlargest(2, "PTS")
ajouter("M11", "Quels sont les deux meilleurs marqueurs de l'équipe OKC (Oklahoma City Thunder) sur la saison ?",
        f"Les deux meilleurs marqueurs d'OKC : {liste(top, 'PTS', ' points')}.", "multicriteres", "excel")

top = df.nlargest(1, "BLK").iloc[0]
ajouter("M12", "Quel joueur a réalisé le plus de contres sur la saison, et pour quelle équipe joue-t-il ?",
        f"{top['Player']} a réalisé le plus de contres ({top['BLK']}). Il joue pour {top['Team']} "
        f"({NOM_EQUIPE[top['Team']]}).", "multicriteres", "excel")

# ---------------------------------------------------------------------------
# 5. Questions BRUITÉES (8) — variantes d'une question propre (fautes, abréviations, anglais)
#    La référence est celle de la question d'origine : seule la formulation change.
# ---------------------------------------------------------------------------
bruitees = [
    ("B01", "S01", "booker 3pt% ??"),
    ("B02", "S07", "herro cb de pts par match en moyene"),
    ("B03", "S04", "gobert il joue ou"),
    ("B04", "S10", "How many three-pointers does Harden attempt per game?"),
    ("B05", "S03", "OFFRTG steph curry"),
    ("B06", "M01", "top 3 shooteurs a 3pts mini 300 tentatives"),
    ("B07", "M02", "qui a mis le + de points cette saison"),
    ("B08", "M03", "meilleurs rebondeurs total saison top3"),
]
for id_, origine, question in bruitees:
    ajouter(id_, question, references[origine], "bruitee", "excel", variante_de=origine)

# ---------------------------------------------------------------------------
# 6. Questions TEXTUELLES (8) — discussions Reddit (r/nba, juin 2025), 2 par fichier
#    Références écrites à la main après lecture des PDF : elles restituent ce que dit la source
#    (ce sont des opinions de fans, pas des faits établis).
# ---------------------------------------------------------------------------
ajouter("T01", "Dans la discussion Reddit sur les équipes qui ont impressionné en playoffs, quels joueurs d'Orlando l'auteur met-il en avant ?",
        "L'auteur met en avant Paolo Banchero et Franz Wagner, qu'il décrit comme son duo de jeunes ailiers préféré, "
        "taillé pour les playoffs. Malgré une mauvaise adresse, Orlando a poussé le champion en titre dans ses retranchements ; "
        "il souhaite que l'équipe recrute de vrais shooteurs et un vrai meneur.",
        "textuelle", "reddit")
ajouter("T02", "Selon les commentaires Reddit, quel joueur des Timberwolves a été une révélation en playoffs, et pourquoi ?",
        "Julius Randle : les commentateurs saluent son jeu physique en attaque (« bully ball »), ses bonnes lectures face aux "
        "prises à deux et son engagement en défense. Il dépasse sa réputation de joueur qui abuse des « spin moves » (la blague « beyblade »).",
        "textuelle", "reddit")
ajouter("T03", "Pourquoi l'auteur d'un post Reddit s'étonne-t-il qu'une finale entre les deux meilleures équipes soit jugée ennuyeuse ?",
        "Il s'étonne qu'une finale potentielle entre les deux meilleures équipes des playoffs selon les statistiques avancées "
        "soit considérée comme sans intérêt, et en accuse les médias NBA, jugés biaisés, ainsi que le marketing de la ligue, "
        "incapable d'intéresser le grand public.",
        "textuelle", "reddit")
ajouter("T04", "Quelle critique est faite à Shai Gilgeous-Alexander dans les commentaires Reddit ?",
        "Certains commentateurs lui reprochent de simuler et de provoquer les fautes (« flopping »), en le comparant à Embiid, "
        "tout en reconnaissant qu'il fait partie des 3 à 5 meilleurs joueurs de la ligue.",
        "textuelle", "reddit")
ajouter("T05", "Selon une discussion Reddit, quel joueur est la première option offensive la plus efficace de l'histoire des playoffs, et sur quels critères ?",
        "Reggie Miller, selon une analyse fondée sur le total de points en playoffs et l'efficacité au tir relative à son époque "
        "(true shooting relatif), comparée à 20 des meilleures premières options de l'histoire.",
        "textuelle", "reddit")
ajouter("T06", "Dans la discussion Reddit sur Reggie Miller, quel joueur est le seul à s'approcher de son efficacité ?",
        "Kawhi Leonard est présenté comme le seul joueur qui s'approche vraiment de l'efficacité de Reggie Miller.",
        "textuelle", "reddit")
ajouter("T07", "Quelle question pose l'auteur du post Reddit sur l'avantage du terrain, et à quelle équipe de hockey fait-il référence ?",
        "Il demande quelle équipe NBA a atteint les Finales sans avoir eu l'avantage du terrain lors des trois premiers tours, "
        "en référence aux Edmonton Oilers en NHL, qui ont joué ces trois tours sans l'avantage du terrain avant de l'avoir en finale.",
        "textuelle", "reddit")
ajouter("T08", "Quelle est la réponse la plus votée à la question de l'avantage du terrain jusqu'aux Finales NBA ?",
        "Selon la réponse la plus votée, six équipes ont atteint les Finales en étant classées au-delà de la 4e place, et aucune "
        "n'a eu l'avantage du terrain en finale : la réponse serait donc aucune équipe, au moins dans la NBA moderne.",
        "textuelle", "reddit")

# ---------------------------------------------------------------------------
# 7. Questions HORS DONNÉES (6) — sur la NBA, mais absentes des données
#    Comportement attendu : signaler que l'information n'est pas disponible, sans l'inventer.
# ---------------------------------------------------------------------------
NON_DISPO = "Cette information n'est pas disponible dans les données : "
ajouter("H01", "Quel joueur a le meilleur pourcentage de réussite à 3 points sur les 5 derniers matchs ?",
        NON_DISPO + "elles ne contiennent que des statistiques cumulées sur la saison régulière, pas de résultats match par match.",
        "hors_donnees", "aucune")
ajouter("H02", "Compare les statistiques de rebonds de l'équipe à domicile et à l'extérieur.",
        NON_DISPO + "elles ne distinguent pas les matchs à domicile et à l'extérieur.",
        "hors_donnees", "aucune")
ajouter("H03", "Combien de points Shai Gilgeous-Alexander a-t-il marqués pendant les playoffs 2025 ?",
        NON_DISPO + "elles ne couvrent que la saison régulière, pas les playoffs.",
        "hors_donnees", "aucune")
ajouter("H04", "Quel est le salaire de Stephen Curry ?",
        NON_DISPO + "elles ne contiennent aucune information contractuelle ou salariale.",
        "hors_donnees", "aucune")
ajouter("H05", "Comment a évolué le pourcentage à 3 points de Stephen Curry par rapport à la saison précédente ?",
        NON_DISPO + "elles ne couvrent qu'une seule saison, sans historique.",
        "hors_donnees", "aucune")
ajouter("H06", "Quelle est la taille de Victor Wembanyama ?",
        NON_DISPO + "elles ne contiennent pas de caractéristiques physiques des joueurs.",
        "hors_donnees", "aucune")

# ---------------------------------------------------------------------------
# 8. Questions HORS SUJET (4) — hors du périmètre de l'assistant (NBA)
#    Comportement attendu : indiquer que la question sort de son périmètre.
# ---------------------------------------------------------------------------
HORS_PERIMETRE = ("Cette question ne concerne pas la NBA : elle sort du périmètre de l'assistant, "
                  "qui doit l'indiquer au lieu d'y répondre.")
for id_, question in [
    ("O01", "Qui a gagné le tournoi de Roland-Garros en 2025 ?"),
    ("O02", "Quelle est la recette de la pâte à crêpes ?"),
    ("O03", "Quelle est la capitale de l'Australie ?"),
    ("O04", "Quelle équipe a remporté la Ligue des champions de football en 2025 ?"),
]:
    ajouter(id_, question, HORS_PERIMETRE, "hors_sujet", "aucune")

# ---------------------------------------------------------------------------
# 9. Contrôles puis écriture du JSON
# ---------------------------------------------------------------------------
ids = [q["id"] for q in testset]
assert len(ids) == len(set(ids)), "Identifiants en double"
assert all(q["ground_truth"].strip() for q in testset), "Référence vide"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(testset, f, ensure_ascii=False, indent=2)

print(f"{len(testset)} questions écrites dans {OUTPUT}")
for categorie, n in Counter(q["category"] for q in testset).items():
    print(f"  {categorie:<14} {n}")
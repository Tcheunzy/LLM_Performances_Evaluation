# load_excel_to_db.py — charge l'Excel « regular NBA » dans une base SQLite (étape 2).
#
# 1. Lecture des feuilles « Equipe » et « Données NBA », validation Pydantic de chaque ligne.
# 2. Création d'une base neuve (tables, contraintes, vue) : schéma dans utils/db_schema.py.
# 3. Insertion dans une seule transaction, puis contrôles (comptes, moyennes de la vue).
# 4. Journal qualité dans docs/audit/anomalies_donnees.csv (rejets, corrections, anomalies conservées).
# Lancement : uv run load_excel_to_db.py
import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import Engine, func, insert, select, text

from utils.db_schema import CHAMPS_HORS_STATS, creer_base, nom_sql, player_stats, players, teams
from utils.schemas import PlayerStats, Team


ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "inputs" / "regular NBA.xlsx"
FEUILLE_JOUEURS = "Données NBA"
FEUILLE_EQUIPES = "Equipe"
DB_PATH = ROOT / "data" / "nba.db"
JOURNAL_CSV = ROOT / "docs" / "audit" / "anomalies_donnees.csv"
logger = logging.getLogger("load_excel_to_db")


def lire_equipes(excel: pd.ExcelFile) -> list[Team]:
    """Feuille « Equipe » -> liste d'équipes validées. Une équipe invalide arrête tout :
    sans table des équipes fiable, aucune clé étrangère ne peut être vérifiée."""
    lignes = excel.parse(FEUILLE_EQUIPES).to_dict("records")
    equipes = [Team.model_validate(ligne) for ligne in lignes]
    codes = [e.code for e in equipes]
    if len(codes) != len(set(codes)):
        raise ValueError("Codes d'équipe en double dans la feuille « Equipe ».")
    return equipes


def lire_joueurs(excel: pd.ExcelFile, codes_equipes: set[str]) -> tuple[list[PlayerStats], list[dict]]:
    """Feuille « Données NBA » -> (joueurs validés, journal).
    Le journal contient une ligne par événement de qualité, avec le traitement appliqué :
      - « rejeté »   : la ligne viole une règle bloquante, le joueur n'est pas chargé ;
      - « corrigé »  : le nom a été nettoyé (virgules -> points) ;
      - « conservé » : anomalie non bloquante, la valeur source est gardée.
    Ce journal deviendra docs/audit/anomalies_donnees.csv (partie 2)."""
    df = excel.parse(FEUILLE_JOUEURS, header=1)                        # vrais en-têtes sur la 2e ligne
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    colonne_heure = [c for c in df.columns if not isinstance(c, str)]  # « 3PM » converti en 15:00:00 par Excel
    df = df.rename(columns={colonne_heure[0]: "3PM"})

    joueurs, journal = [], []
    for ligne in df.to_dict("records"):
        nom_source = ligne.get("Player")
        try:
            joueur = PlayerStats.model_validate(ligne, context={"codes_equipes": codes_equipes})
        except ValidationError as e:
            journal.append({"joueur": nom_source, "regle": "validation", "detail": e.errors()[0]["msg"],
                            "traitement": "rejeté"})
            continue
        if joueur.player != nom_source:
            journal.append({"joueur": joueur.player, "regle": "nom", "detail": f"{nom_source!r} -> {joueur.player!r}",
                            "traitement": "corrigé"})
        for anomalie in joueur.anomalies:
            journal.append({"joueur": joueur.player, "regle": "cohérence", "detail": anomalie,
                            "traitement": "conservé"})
        joueurs.append(joueur)

    noms = [j.player for j in joueurs]
    doublons = {n for n in noms if noms.count(n) > 1}
    if doublons:
        raise ValueError(f"Noms en double après nettoyage (contrainte UNIQUE impossible) : {doublons}")
    return joueurs, journal


def inserer(moteur: Engine, equipes: list[Team], joueurs: list[PlayerStats]) -> None:
    """Insère équipes, joueurs et statistiques dans UNE transaction : tout ou rien."""
    lignes_teams = [{"code": e.code, "name": e.name} for e in equipes]
    lignes_players, lignes_stats = [], []
    for player_id, j in enumerate(joueurs, start=1):
        lignes_players.append({"player_id": player_id, "name": j.player, "team_code": j.team, "age": j.age})
        stats = j.model_dump(exclude=CHAMPS_HORS_STATS)
        lignes_stats.append({"player_id": player_id, **{nom_sql(c): v for c, v in stats.items()}})

    with moteur.begin() as connexion:  # ordre imposé par les clés étrangères : teams -> players -> stats
        connexion.execute(insert(teams), lignes_teams)
        connexion.execute(insert(players), lignes_players)
        connexion.execute(insert(player_stats), lignes_stats)


def controler(moteur: Engine, equipes: list[Team], joueurs: list[PlayerStats]) -> None:
    """Relit la base : les comptes doivent correspondre, et la vue doit donner les moyennes attendues."""
    with moteur.connect() as connexion:
        comptes = {t.name: connexion.execute(select(func.count()).select_from(t)).scalar_one()
                   for t in (teams, players, player_stats)}
        exemples = connexion.execute(text(
            "SELECT name, pts_per_game, three_pa_per_game FROM player_per_game "
            "WHERE name IN ('Shai Gilgeous-Alexander', 'Tyler Herro', 'James Harden') ORDER BY name"
        )).all()
    attendus = {"teams": len(equipes), "players": len(joueurs), "player_stats": len(joueurs)}
    for table, n in comptes.items():
        statut = "OK" if n == attendus[table] else f"ERREUR (attendu {attendus[table]})"
        print(f"  {table:<13}: {n} lignes -> {statut}")
    print("  Vue player_per_game :")
    for nom, pts, three_pa in exemples:
        print(f"    {nom:<24} {pts} pts/match, {three_pa} tirs à 3 pts tentés/match")


def controler(moteur: Engine, equipes: list[Team], joueurs: list[PlayerStats]) -> bool:
    """Relit la base : les comptes doivent correspondre, et la vue doit donner les moyennes attendues.
    Renvoie True si tous les comptes sont bons."""
    with moteur.connect() as connexion:
        comptes = {t.name: connexion.execute(select(func.count()).select_from(t)).scalar_one()
                   for t in (teams, players, player_stats)}
        exemples = connexion.execute(text(
            "SELECT name, pts_per_game, three_pa_per_game FROM player_per_game "
            "WHERE name IN ('Shai Gilgeous-Alexander', 'Tyler Herro', 'James Harden') ORDER BY name"
        )).all()
    attendus = {"teams": len(equipes), "players": len(joueurs), "player_stats": len(joueurs)}
    ok = True
    for table, n in comptes.items():
        if n == attendus[table]:
            logger.info("Contrôle %s : %d lignes (OK)", table, n)
        else:
            logger.error("Contrôle %s : %d lignes, %d attendues", table, n, attendus[table])
            ok = False
    for nom, pts, three_pa in exemples:
        logger.info("Vue player_per_game : %s = %s pts/match, %s tirs à 3 pts tentés/match", nom, pts, three_pa)
    return ok


def charger_base(excel_path: Path = EXCEL, db_path: Path = DB_PATH, journal_csv: Path = JOURNAL_CSV) -> dict:
    """Étapes complètes : lecture, validation, base neuve, insertion, contrôle, journal qualité.
    Renvoie un résumé (pour setup_rag.py) ; lève RuntimeError si le contrôle échoue."""
    excel = pd.ExcelFile(excel_path)
    equipes = lire_equipes(excel)
    joueurs, journal = lire_joueurs(excel, {e.code for e in equipes})
    resume = {traitement: sum(l["traitement"] == traitement for l in journal)
              for traitement in ["rejeté", "corrigé", "conservé"]}
    logger.info("Validation : %d équipes, %d joueurs valides, %d rejetés, %d corrigés, %d anomalies conservées",
                len(equipes), len(joueurs), resume["rejeté"], resume["corrigé"], resume["conservé"])
    for l in journal:
        if l["traitement"] == "rejeté":
            logger.warning("Joueur rejeté : %s (%s)", l["joueur"], l["detail"])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # base reconstruite à chaque lancement : le script est rejouable
    moteur = creer_base(db_path)
    inserer(moteur, equipes, joueurs)
    logger.info("Base créée : %s", db_path)
    ok = controler(moteur, equipes, joueurs)
    moteur.dispose()  # libère le fichier (sinon Windows peut le garder verrouillé)

    journal_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(journal, columns=["joueur", "regle", "detail", "traitement"]).to_csv(
        journal_csv, index=False, encoding="utf-8-sig")
    logger.info("Journal qualité (%d lignes) : %s", len(journal), journal_csv)
    if not ok:
        raise RuntimeError("Les comptes de la base ne correspondent pas aux données validées.")
    return {"equipes": len(equipes), "joueurs": len(joueurs), **resume}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    charger_base()
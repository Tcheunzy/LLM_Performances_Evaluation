# utils/db_schema.py — schéma de la base SQLite (étape 2) : tables, contraintes et vue.
#
# Les colonnes de player_stats sont GÉNÉRÉES à partir du modèle Pydantic PlayerStats :
# une seule source de vérité. Ajouter un champ au modèle ajoute la colonne à la base.
# Le module est importé par load_excel_to_db.py (création) et servira à sql_tool.py (description).
from sqlalchemy import (CheckConstraint, Column, Float, ForeignKey, Integer, MetaData, String, Table,
                        create_engine, event, text)
from sqlalchemy.engine import Engine

from utils.schemas import PlayerStats

metadata = MetaData()

# Champs du modèle qui ne vont PAS dans player_stats (identité -> table players ; anomalies -> CSV d'audit)
CHAMPS_HORS_STATS = {"player", "team", "age", "anomalies"}
# Noms de champs Pydantic renommés en SQL, pour que le LLM comprenne la colonne sans contexte
RENOMMAGE_SQL = {"w": "wins", "l": "losses"}
TYPES_SQL = {int: Integer, float: Float}

# Totaux de saison divisés par gp dans la vue player_per_game
TOTAUX_PAR_MATCH = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
                    "fgm", "fga", "three_pm", "three_pa", "ftm", "fta"]


def nom_sql(champ: str) -> str:
    return RENOMMAGE_SQL.get(champ, champ)


teams = Table(
    "teams", metadata,
    Column("code", String(3), primary_key=True),
    Column("name", String, nullable=False, unique=True),
)

players = Table(
    "players", metadata,
    Column("player_id", Integer, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("team_code", String(3), ForeignKey("teams.code"), nullable=False),
    Column("age", Integer, nullable=False),
)

player_stats = Table(
    "player_stats", metadata,
    Column("player_id", Integer, ForeignKey("players.player_id"), primary_key=True),
    *[
        Column(nom_sql(champ), TYPES_SQL[info.annotation], nullable=False)
        for champ, info in PlayerStats.model_fields.items()
        if champ not in CHAMPS_HORS_STATS
    ],
    CheckConstraint("wins + losses = gp", name="ck_wins_losses_gp"),
)

# Table prévue pour des données match par match : vide, car l'Excel ne contient que des totaux de saison.
matches = Table(
    "matches", metadata,
    Column("match_id", Integer, primary_key=True),
    Column("season", String, nullable=False),
    Column("match_date", String, nullable=False),  # date ISO 'AAAA-MM-JJ' (SQLite n'a pas de type date)
    Column("home_team_code", String(3), ForeignKey("teams.code"), nullable=False),
    Column("away_team_code", String(3), ForeignKey("teams.code"), nullable=False),
    Column("home_score", Integer),
    Column("away_score", Integer),
    CheckConstraint("home_team_code <> away_team_code", name="ck_equipes_differentes"),
    CheckConstraint("home_score >= 0 AND away_score >= 0", name="ck_scores_positifs"),
)

# Vue : moyennes par match calculées par la base (et non par le LLM). « * 1.0 » évite la division entière.
VUE_PAR_MATCH = (
    "CREATE VIEW player_per_game AS\n"
    "SELECT p.name, p.team_code, t.name AS team_name, s.gp, s.min_per_game,\n  "
    + ",\n  ".join(f"ROUND(s.{c} * 1.0 / s.gp, 1) AS {c}_per_game" for c in TOTAUX_PAR_MATCH)
    + "\nFROM players p\n"
    "JOIN player_stats s ON s.player_id = p.player_id\n"
    "JOIN teams t ON t.code = p.team_code"
)


@event.listens_for(Engine, "connect")
def activer_cles_etrangeres(connexion_dbapi, _):
    """SQLite n'applique pas les clés étrangères par défaut : on les active à chaque connexion."""
    curseur = connexion_dbapi.cursor()
    curseur.execute("PRAGMA foreign_keys = ON")
    curseur.close()


def creer_base(chemin_db) -> Engine:
    """Crée une base neuve (tables + vue) et renvoie le moteur SQLAlchemy."""
    moteur = create_engine(f"sqlite:///{chemin_db}")
    metadata.create_all(moteur)
    with moteur.begin() as connexion:
        connexion.execute(text(VUE_PAR_MATCH))
    return moteur
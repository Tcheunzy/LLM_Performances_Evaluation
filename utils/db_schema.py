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

# Description de chaque colonne, avec son UNITÉ : c'est ce texte que lit le LLM qui écrit le SQL.
# (Le dictionnaire de l'Excel est faux sur les unités : PTS, FGM... y sont dits « par match », ce sont des totaux.)
DESCRIPTIONS = {
    "teams": {
        "code": "code de l'équipe à 3 lettres, ex. 'OKC' (clé primaire)",
        "name": "nom complet de l'équipe, ex. 'Oklahoma City Thunder'",
    },
    "players": {
        "player_id": "identifiant du joueur (clé primaire)",
        "name": "nom du joueur, ex. 'Shai Gilgeous-Alexander' (peut contenir des accents : 'Nikola Jokić')",
        "team_code": "code de l'équipe (clé étrangère vers teams.code)",
        "age": "âge en années",
    },
    "player_stats": {
        "player_id": "identifiant du joueur (clé étrangère vers players.player_id)",
        "gp": "nombre de matchs joués",
        "wins": "victoires de l'équipe lors des matchs joués par le joueur",
        "losses": "défaites de l'équipe lors des matchs joués par le joueur",
        "pts": "points marqués, TOTAL saison",
        "fgm": "tirs réussis, TOTAL saison", "fga": "tirs tentés, TOTAL saison",
        "three_pm": "tirs à 3 points réussis, TOTAL saison", "three_pa": "tirs à 3 points tentés, TOTAL saison",
        "ftm": "lancers francs réussis, TOTAL saison", "fta": "lancers francs tentés, TOTAL saison",
        "oreb": "rebonds offensifs, TOTAL saison", "dreb": "rebonds défensifs, TOTAL saison",
        "reb": "rebonds, TOTAL saison", "ast": "passes décisives, TOTAL saison",
        "tov": "balles perdues, TOTAL saison", "stl": "interceptions, TOTAL saison",
        "blk": "contres, TOTAL saison", "pf": "fautes personnelles, TOTAL saison",
        "fantasy_pts": "points fantasy, TOTAL saison",
        "double_doubles": "nombre de double-doubles", "triple_doubles": "nombre de triple-doubles",
        "poss": "possessions jouées, TOTAL saison",
        "min_per_game": "minutes jouées, MOYENNE par match",
        "plus_minus": "plus/minus, MOYENNE par match",
        "fg_pct": "% de réussite aux tirs, de 0 à 100 (45.2 = 45,2 %)",
        "three_pct": "% de réussite à 3 points, de 0 à 100",
        "ft_pct": "% de réussite aux lancers francs, de 0 à 100",
        "ast_pct": "% des paniers de l'équipe sur passe du joueur, de 0 à 100",
        "oreb_pct": "% des rebonds offensifs disponibles captés, de 0 à 100",
        "dreb_pct": "% des rebonds défensifs disponibles captés, de 0 à 100",
        "reb_pct": "% des rebonds disponibles captés, de 0 à 100",
        "usg_pct": "taux d'utilisation (% des actions de l'équipe), de 0 à 100",
        "efg_pct": "effective field goal %, de 0 à 100 (peut dépasser 100 sur très peu de tirs)",
        "ts_pct": "true shooting %, de 0 à 100 (peut dépasser 100 sur très peu de tirs)",
        "off_rating": "offensive rating : points marqués par l'équipe pour 100 possessions",
        "def_rating": "defensive rating : points encaissés pour 100 possessions (plus bas = meilleur)",
        "net_rating": "net rating = off_rating - def_rating",
        "ast_to_ratio": "passes décisives par balle perdue",
        "ast_ratio": "passes décisives pour 100 possessions",
        "tov_ratio": "balles perdues pour 100 possessions",
        "pace": "rythme : possessions par 48 minutes",
        "pie": "PIE (Player Impact Estimate), indice d'impact global",
    },
    "matches": {
        "match_id": "identifiant du match", "season": "saison, ex. '2024-25'",
        "match_date": "date au format 'AAAA-MM-JJ'",
        "home_team_code": "équipe à domicile", "away_team_code": "équipe à l'extérieur",
        "home_score": "score à domicile", "away_score": "score à l'extérieur",
    },
}

NOTES_TABLES = {
    "player_stats": "une ligne par joueur pour TOUTE la saison régulière 2024-25 ; "
                    "il n'y a PAS de colonne season ni de données par match ou par période",
    "matches": "ATTENTION : table VIDE. Les données fournies sont des totaux de saison, "
               "il n'existe aucun résultat match par match.",
}

def decrire_schema() -> str:
    """Description textuelle de la base pour le prompt du LLM (tables, colonnes, unités, vue)."""
    blocs = []
    for table in metadata.sorted_tables:
        colonnes = "\n".join(f"  - {c.name} : {DESCRIPTIONS[table.name][c.name]}" for c in table.columns)
        note = f" ({NOTES_TABLES[table.name]})" if table.name in NOTES_TABLES else ""
        blocs.append(f"Table {table.name}{note} :\n{colonnes}")
    colonnes_vue = ", ".join(f"{c}_per_game" for c in TOTAUX_PAR_MATCH)
    blocs.append(
        "Vue player_per_game (MOYENNES PAR MATCH déjà calculées, arrondies à 0,1) :\n"
        f"  - name, team_code, team_name, gp, min_per_game, {colonnes_vue}"
    )
    return "\n\n".join(blocs)


# Garde-fou : une colonne ajoutée au modèle sans description ferait planter decrire_schema() plus tard.
_manquantes = [f"{t.name}.{c.name}" for t in metadata.sorted_tables for c in t.columns
               if c.name not in DESCRIPTIONS.get(t.name, {})]
if _manquantes:
    raise RuntimeError(f"Colonnes sans description dans DESCRIPTIONS : {_manquantes}")
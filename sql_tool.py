# sql_tool.py — outil SQL de l'assistant (étape 2).
#
# Brique 1 : exécution SÛRE d'une requête écrite par un LLM.
#   - connexion en LECTURE SEULE : même une requête malveillante ne peut rien modifier ;
#   - garde-fou : une seule instruction, SELECT ou WITH uniquement, mots-clés d'écriture refusés ;
#   - résultat borné (MAX_LIGNES) et erreurs renvoyées sous forme de texte, pour que le LLM se corrige.
# Brique 2 : génération de la requête par le LLM (LangChain, prompt few-shot), validée par Pydantic,
#   avec une tentative de correction si SQLite renvoie une erreur.
# Lancement du test : uv run sql_tool.py
import re
import sqlite3
import unicodedata
from pathlib import Path
from sqlalchemy.pool import NullPool
import logfire
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from utils.config import MISTRAL_API_KEY, MODEL_NAME
from utils.db_schema import decrire_schema  # importer le module active aussi le PRAGMA foreign_keys

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "nba.db"
MAX_LIGNES = 50
MAX_TENTATIVES = 2  # 1 génération + 1 correction

MOTS_INTERDITS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|TRUNCATE)\b",
    re.IGNORECASE,
)


# =====================================================================================
# Brique 1 : exécution sûre
# =====================================================================================
def sans_accents(texte: str | None) -> str | None:
    """« Nikola Jokić » -> « nikola jokic ». Enregistrée comme fonction SQL (SQLite n'a pas d'ILIKE)."""
    if texte is None:
        return None
    texte = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c)).lower()


def _connexion_lecture_seule() -> sqlite3.Connection:
    """Ouvre le fichier en mode « ro » (read-only) : SQLite refuse alors toute écriture."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Base introuvable : {DB_PATH}. Lancer d'abord : uv run load_excel_to_db.py")
    connexion = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    connexion.create_function("sans_accents", 1, sans_accents, deterministic=True)
    return connexion


# NullPool : une connexion neuve par requête, ouverte et fermée dans le même thread.
# (Pydantic AI exécute les outils dans un thread séparé ; SQLite refuse qu'une connexion change de thread.)
moteur = create_engine("sqlite://", creator=_connexion_lecture_seule, poolclass=NullPool)


def verifier_requete(sql: str) -> str:
    """Renvoie la requête nettoyée si elle est autorisée, lève ValueError sinon."""
    requete = sql.strip().rstrip(";").strip()
    if not requete:
        raise ValueError("Requête vide.")
    if ";" in requete:
        raise ValueError("Une seule instruction SQL est autorisée.")
    if not re.match(r"^(SELECT|WITH)\b", requete, re.IGNORECASE):
        raise ValueError("Seules les requêtes SELECT (ou WITH ... SELECT) sont autorisées.")
    interdit = MOTS_INTERDITS.search(requete)
    if interdit:
        raise ValueError(f"Mot-clé interdit : {interdit.group(0).upper()}.")
    return requete


def executer_requete(sql: str) -> dict:
    """Exécute une requête de lecture. Ne lève jamais d'exception : l'erreur est renvoyée en texte,
    pour que l'agent puisse la lire et corriger sa requête."""
    try:
        requete = verifier_requete(sql)
        with moteur.connect() as connexion:
            resultat = connexion.execute(text(requete))
            colonnes = list(resultat.keys())
            lignes = [list(ligne) for ligne in resultat.fetchmany(MAX_LIGNES + 1)]
    except (ValueError, SQLAlchemyError) as e:
        message = str(getattr(e, "orig", None) or e)  # message SQLite d'origine, plus lisible
        return {"requete": sql, "erreur": message}
    return {
        "requete": requete,
        "colonnes": colonnes,
        "lignes": lignes[:MAX_LIGNES],
        "tronque": len(lignes) > MAX_LIGNES,
    }


# =====================================================================================
# Brique 2 : génération de la requête par le LLM (LangChain, few-shot)
# =====================================================================================
SYSTEM_SQL = """Tu traduis des questions sur la saison régulière NBA en UNE requête SQL pour SQLite.

Schéma de la base :
{schema}

Règles :
1. Réponds UNIQUEMENT par la requête SQL, sans explication ni balises.
2. Uniquement SELECT (ou WITH ... SELECT). Jamais d'écriture.
3. Les colonnes de player_stats sont des TOTAUX de saison. Pour une moyenne « par match »,
   utilise la vue player_per_game au lieu de diviser toi-même.
4. Les pourcentages vont de 0 à 100 : « plus de 40 % » s'écrit three_pct > 40.
5. Pour retrouver un joueur, utilise sans_accents(name) LIKE '%...%' en minuscules sans accents,
   car l'orthographe de la question peut différer de la base (accents, prénom absent).
6. Mets toujours le nom du joueur (et la valeur demandée) dans le SELECT, et ajoute LIMIT 20
   si la question ne précise pas le nombre de résultats.
7. Une équipe peut être désignée par son code (teams.code, 'OKC') ou son nom (teams.name).
8. Pour classer des joueurs sur un pourcentage ou un ratio SANS minimum précisé dans la question,
   ajoute un seuil de volume raisonnable (ft_pct : fta >= 50 ; three_pct : three_pa >= 100 ;
   fg_pct, efg_pct, ts_pct : fga >= 200) et mets la colonne de volume dans le SELECT.
"""

# Exemples few-shot : ils enseignent des MOTIFS (jointure, vue, filtre, agrégation...).
# Aucun n'est une question du jeu de test (eval/testset.json), sinon l'évaluation v2 serait biaisée.
EXEMPLES_FEW_SHOT = [
    {"question": "Combien d'interceptions Anthony Edwards a-t-il réalisées cette saison ?",
     "requete": "SELECT p.name, s.stl FROM players p JOIN player_stats s ON s.player_id = p.player_id "
                "WHERE sans_accents(p.name) LIKE '%anthony edwards%'"},
    {"question": "Combien de passes décisives par match pour Doncic ?",
     "requete": "SELECT name, ast_per_game FROM player_per_game WHERE sans_accents(name) LIKE '%doncic%'"},
    {"question": "Quels sont les 5 meilleurs intercepteurs parmi les joueurs ayant disputé au moins 50 matchs ?",
     "requete": "SELECT p.name, p.team_code, s.stl, s.gp FROM players p "
                "JOIN player_stats s ON s.player_id = p.player_id "
                "WHERE s.gp >= 50 ORDER BY s.stl DESC LIMIT 5"},
    {"question": "Quels joueurs ont un true shooting supérieur à 65 % avec au moins 500 tirs tentés ?",
     "requete": "SELECT p.name, s.ts_pct, s.fga FROM players p JOIN player_stats s ON s.player_id = p.player_id "
                "WHERE s.ts_pct > 65 AND s.fga >= 500 ORDER BY s.ts_pct DESC LIMIT 20"},
    {"question": "Qui est le meilleur passeur par match des Denver Nuggets ?",
     "requete": "SELECT name, team_name, ast_per_game FROM player_per_game "
                "WHERE team_name = 'Denver Nuggets' ORDER BY ast_per_game DESC LIMIT 1"},
    {"question": "Compare les points par match de Kevin Durant et LeBron James.",
     "requete": "SELECT name, gp, pts_per_game FROM player_per_game "
                "WHERE sans_accents(name) LIKE '%kevin durant%' OR sans_accents(name) LIKE '%lebron james%'"},
    {"question": "Quelle équipe compte le plus de joueurs à au moins 1000 points ?",
     "requete": "SELECT t.name, COUNT(*) AS nb_joueurs FROM players p "
                "JOIN player_stats s ON s.player_id = p.player_id JOIN teams t ON t.code = p.team_code "
                "WHERE s.pts >= 1000 GROUP BY t.name ORDER BY nb_joueurs DESC LIMIT 5"},
    {"question": "Quel était le score du match Lakers - Celtics du 25 décembre ?",
     "requete": "SELECT match_date, home_team_code, away_team_code, home_score, away_score FROM matches "
                "WHERE match_date LIKE '%-12-25' AND home_team_code IN ('LAL', 'BOS') "
                "AND away_team_code IN ('LAL', 'BOS')"},
    {"question": "Quel joueur des Los Angeles Lakers a le meilleur pourcentage de réussite aux tirs ?",
     "requete": "SELECT p.name, s.fg_pct, s.fga FROM players p "
                "JOIN player_stats s ON s.player_id = p.player_id JOIN teams t ON t.code = p.team_code "
                "WHERE t.name = 'Los Angeles Lakers' AND s.fga >= 200 ORDER BY s.fg_pct DESC LIMIT 1"},
]


def extraire_sql(texte: str) -> str:
    """Isole la requête dans la réponse brute du LLM.

    Les petits modèles ajoutent souvent du texte autour de la requête malgré la consigne :
    balises ```sql, explications, commentaires « -- ». Si la réponse contient un ou plusieurs blocs
```sql ... ```, on garde le DERNIER (c'est en général la version corrigée)."""
    texte = str(texte).strip()
    blocs = re.findall(r"```(?:sql)?\s*(.*?)```", texte, flags=re.IGNORECASE | re.DOTALL)
    if blocs:
        texte = blocs[-1]
    elif "```" in texte:  # bloc ouvert mais jamais refermé
        texte = re.sub(r"```(?:sql)?", "", texte, flags=re.IGNORECASE)
    # Commentaires SQL « -- ... » (pas les « -- » collés à du texte, ex. dans une chaîne 'a--b')
    texte = re.sub(r"(^|\s)--(\s|$)[^\n]*", r"\1", texte, flags=re.MULTILINE)
    return texte.strip()


class RequeteSQL(BaseModel):
    """Requête produite par le LLM. La validation Pydantic applique le garde-fou de la brique 1."""

    requete: str = Field(min_length=10)

    @field_validator("requete", mode="before")
    @classmethod
    def nettoyer_et_verifier(cls, valeur: str) -> str:
        return verifier_requete(extraire_sql(valeur))



prompt_exemple = ChatPromptTemplate.from_messages([("human", "{question}"), ("ai", "{requete}")])
prompt_sql = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_SQL),
    FewShotChatMessagePromptTemplate(examples=EXEMPLES_FEW_SHOT, example_prompt=prompt_exemple),
    ("human", "{question}"),
]).partial(schema=decrire_schema())

# max_tokens : une requête tient en quelques lignes ; on coupe court aux réponses qui s'emballent
# (une génération de 2 minutes a été observée). timeout : une requête HTTP bloquée est relancée vite.
llm_sql = ChatMistralAI(model=MODEL_NAME, api_key=MISTRAL_API_KEY, temperature=0, max_retries=5,
                        max_tokens=400, timeout=30)
chaine_sql = prompt_sql | llm_sql | StrOutputParser()


def interroger_base(question: str) -> dict:
    """Question en langage naturel -> requête SQL générée, validée, exécutée.

    Si la requête est refusée (garde-fou) ou échoue (erreur SQLite), le LLM reçoit l'erreur et
    a droit à MAX_TENTATIVES - 1 correction. Renvoie le dictionnaire de executer_requete(),
    enrichi de la question et du nombre de tentatives."""
    with logfire.span("sql_tool", question=question):
        consigne = question
        for tentative in range(1, MAX_TENTATIVES + 1):
            brut = chaine_sql.invoke({"question": consigne})
            try:
                resultat = executer_requete(RequeteSQL(requete=brut).requete)
            except ValidationError as e:
                resultat = {"requete": brut, "erreur": e.errors()[0]["msg"]}
            logfire.info("Tentative {tentative} : {requete}", tentative=tentative,
                         requete=resultat["requete"], erreur=resultat.get("erreur"))
            if "erreur" not in resultat:
                break
            consigne = (f"{question}\n\nTa requête précédente a échoué.\nRequête : {resultat['requete']}\n"
                        f"Erreur : {resultat['erreur']}\nÉcris une requête corrigée.")
        return {"question": question, "tentatives": tentative, **resultat}


if __name__ == "__main__":
    logfire.configure(service_name="sql_tool", send_to_logfire="if-token-present")
    questions = [
        "Combien de points par match marque Jayson Tatum ?",
        "Quels sont les 3 joueurs avec le plus de contres par match parmi ceux ayant joué 40 matchs ?",
        "Quel joueur des Boston Celtics a le meilleur pourcentage aux lancers francs ?",
        "Combien de joueurs ont plus de 35 ans ?",
        "Qui a gagné le match du 12 mars entre OKC et Minnesota ?",
    ]
    for q in questions:
        r = interroger_base(q)
        print(f"\nQ : {q}\nSQL ({r['tentatives']} tentative(s)) : {r['requete']}")
        print(f"-> ERREUR : {r['erreur']}" if "erreur" in r else f"-> {r['colonnes']} {r['lignes'][:5]}")
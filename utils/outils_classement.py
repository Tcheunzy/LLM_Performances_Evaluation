# utils/outils_classement.py — outils de calcul DÉTERMINISTES pour l'agent (v2d).
#
# Pourquoi : l'évaluation des questions mixtes a montré que le SQL généré par le LLM est faux sur les rangs
# (filtre sur le joueur AVANT le classement -> toujours « 1er ») et que le modèle se trompe en comparant
# deux nombres (« 35,6 ne dépasse pas 35 »). Un exemple few-shot et une règle de prompt n'ont pas suffi (v2c).
# Ici, le SQL est écrit à l'avance et testé : le modèle ne choisit que les PARAMÈTRES (statistique, joueur,
# seuil), le code fait le calcul et renvoie une conclusion en clair, que le modèle n'a plus qu'à recopier.
#
# v2e : limite d'appels rendue stricte (reserver_appel, appels parallèles) et nom du joueur nettoyé.
#
# Sécurité : la statistique est un Literal (liste fermée des colonnes de player_stats) -> seul un nom de
# colonne connu est inséré dans le SQL ; le joueur et le seuil passent en paramètres liés (?), jamais
# concaténés. La connexion reste en lecture seule (celle de sql_tool).
from typing import TYPE_CHECKING, Literal

from pydantic_ai import RunContext

from sql_tool import _connexion_lecture_seule, sans_accents
from utils.agent_outils import MAX_APPELS_SQL, SOURCE_SQL, ContexteRecherche, reserver_appel
from utils.db_schema import DESCRIPTIONS
from utils.nettoyage_donnees import nettoyer_nom_joueur

COLONNES_STATS = tuple(c for c in DESCRIPTIONS["player_stats"] if c != "player_id")
if TYPE_CHECKING:
    Statistique = str  # pour l'éditeur (Pylance) : un Literal construit à l'exécution n'est pas analysable
else:
    Statistique = Literal[COLONNES_STATS]  # à l'exécution : énumération dans le schéma JSON envoyé au LLM
LISTE_STATS = "\n".join(f"- {c} : {d}" for c, d in DESCRIPTIONS["player_stats"].items() if c != "player_id")

SQL_CLASSEMENT = """
WITH classement AS (
    SELECT p.name, p.team_code, s.gp, s.{col} AS valeur,
           RANK() OVER (ORDER BY s.{col} {sens}) AS rang,
           COUNT(*) OVER () AS nb_joueurs
    FROM players p JOIN player_stats s ON s.player_id = p.player_id
    WHERE s.gp >= ? AND s.{col} IS NOT NULL
)
SELECT name, team_code, gp, valeur, rang, nb_joueurs FROM classement {filtre} ORDER BY rang, name
"""


def _classement(statistique: str, ordre: str, min_matchs: int, joueur: str | None = None,
                limite: int | None = None) -> list[tuple]:
    """Classement de TOUS les joueurs (ayant au moins min_matchs matchs), PUIS filtrage éventuel sur un joueur."""
    if statistique not in COLONNES_STATS:  # double sécurité : le Literal l'empêche déjà
        raise ValueError(f"Statistique inconnue : {statistique}")
    params: list = [min_matchs]
    filtre = ""
    if joueur:
        filtre = "WHERE sans_accents(name) LIKE ?"
        # Même nettoyage qu'au chargement de la base : « P,J, Washington » (nom abîmé des fiches FAISS) -> « P.J. Washington »
        params.append(f"%{sans_accents(nettoyer_nom_joueur(joueur).lower())}%")
    sql = SQL_CLASSEMENT.format(col=statistique, sens="DESC" if ordre == "decroissant" else "ASC", filtre=filtre)
    if limite:
        sql += f" LIMIT {int(limite)}"
    connexion = _connexion_lecture_seule()
    try:
        return connexion.execute(sql, params).fetchall()
    finally:
        connexion.close()


def _tracer(ctx: RunContext[ContexteRecherche], outil: str, parametres: dict, texte: str, ok: bool) -> str:
    """Même traçabilité que interroger_statistiques : trace, contexte RAGAS et source autorisée."""
    ctx.deps.requetes_sql.append({"question": f"{outil}({parametres})", "requete": f"[requête paramétrée] {outil}",
                                  "tentatives": 1, "nb_lignes": int(ok), "erreur": None if ok else texte})
    texte = f"[Source : {SOURCE_SQL}] Calcul : {outil}({parametres})\n{texte}"
    if ok:
        ctx.deps.contextes.append(texte)
        if SOURCE_SQL not in ctx.deps.sources_autorisees:
            ctx.deps.sources_autorisees.append(SOURCE_SQL)
    return texte


def _identifier(lignes: list[tuple], joueur: str) -> tuple | str:
    """Une seule ligne attendue ; sinon, message explicite (introuvable ou ambigu)."""
    if not lignes:
        return f"Aucun joueur ne correspond à « {joueur} » : l'information n'est pas disponible."
    if len(lignes) > 1:
        noms = ", ".join(l[0] for l in lignes[:10])
        return f"Plusieurs joueurs correspondent à « {joueur} » : {noms}. Précise le nom complet."
    return lignes[0]


def rang_joueur(ctx: RunContext[ContexteRecherche], statistique: Statistique, joueur: str,
                ordre: Literal["decroissant", "croissant"] = "decroissant", min_matchs: int = 0) -> str:
    """Donne la valeur d'UN joueur DÉJÀ NOMMÉ dans la question et son rang parmi TOUS les joueurs de la
    ligue, avec la tête du classement : « quel est son rang ? », « est-ce le meilleur total ? »,
    « se classe-t-il parmi ceux qui... le plus ? ». Recopie le rang renvoyé, ne le déduis jamais toi-même.
    NE PAS utiliser pour trouver QUEL joueur est le meilleur ni pour un top N : c'est le rôle de
    interroger_statistiques. Ne pas l'appeler joueur par joueur sur les fiches des documents.

    Args:
        statistique: colonne de la base (totaux de saison sauf mention contraire) :
{liste}
        joueur: nom complet du joueur, ex. « Shai Gilgeous-Alexander ».
        ordre: « decroissant » (rang 1 = valeur la plus élevée, cas général) ou « croissant »
            (rang 1 = valeur la plus basse, ex. def_rating).
        min_matchs: nombre minimal de matchs joués pour figurer dans le classement (0 par défaut).
    """
    parametres = {"statistique": statistique, "joueur": joueur, "ordre": ordre, "min_matchs": min_matchs}
    if not reserver_appel(ctx):
        return f"Limite de {MAX_APPELS_SQL} interrogations atteinte. Réponds avec les résultats déjà obtenus."
    ligne = _identifier(_classement(statistique, ordre, min_matchs, joueur=joueur), joueur)
    if isinstance(ligne, str):
        return _tracer(ctx, "rang_joueur", parametres, ligne, ok=False)
    nom, equipe, gp, valeur, rang, nb = ligne
    tete = _classement(statistique, ordre, min_matchs, limite=3)
    ex_aequo = sum(1 for l in _classement(statistique, ordre, min_matchs) if l[4] == rang) > 1
    sens = "1 = valeur la plus élevée" if ordre == "decroissant" else "1 = valeur la plus basse"
    if rang == 1:
        verdict = "C'est la meilleure valeur de la ligue (1er)."
    else:
        verdict = f"Ce n'est PAS la meilleure valeur de la ligue : il est {rang}e."
    texte = (f"{nom} ({equipe}, {gp} matchs) : {statistique} = {valeur}. "
             f"Rang : {rang}{' (ex aequo)' if ex_aequo else ''} sur {nb} joueurs ({sens}).\n{verdict}\n"
             "Tête du classement : " + " ; ".join(f"{l[4]}. {l[0]} ({l[3]})" for l in tete))
    return _tracer(ctx, "rang_joueur", parametres, texte, ok=True)


def comparer_a_un_seuil(ctx: RunContext[ContexteRecherche], statistique: Statistique, joueur: str,
                        seuil: float) -> str:
    """Compare la valeur d'un joueur pour une statistique de la saison à un seuil donné et renvoie la
    conclusion. À utiliser pour vérifier une affirmation chiffrée : « a-t-il dépassé 35 % ? »,
    « a-t-il marqué plus de 2000 points ? ». Recopie la conclusion renvoyée telle quelle.

    Args:
        statistique: colonne de la base (mêmes colonnes que pour rang_joueur).
        joueur: nom complet du joueur.
        seuil: valeur de comparaison, dans l'unité de la colonne (35 pour 35 %).
    """
    parametres = {"statistique": statistique, "joueur": joueur, "seuil": seuil}
    if not reserver_appel(ctx):
        return f"Limite de {MAX_APPELS_SQL} interrogations atteinte. Réponds avec les résultats déjà obtenus."
    ligne = _identifier(_classement(statistique, "decroissant", 0, joueur=joueur), joueur)
    if isinstance(ligne, str):
        return _tracer(ctx, "comparer_a_un_seuil", parametres, ligne, ok=False)
    nom, equipe, gp, valeur, _, _ = ligne
    if valeur > seuil:
        conclusion = f"{valeur} est SUPÉRIEUR à {seuil} : OUI, le seuil est dépassé."
    elif valeur < seuil:
        conclusion = f"{valeur} est INFÉRIEUR à {seuil} : NON, le seuil n'est pas atteint."
    else:
        conclusion = f"{valeur} est ÉGAL à {seuil} : le seuil est atteint mais pas dépassé."
    texte = f"{nom} ({equipe}) : {statistique} = {valeur}. {conclusion}"
    return _tracer(ctx, "comparer_a_un_seuil", parametres, texte, ok=True)


# La liste des colonnes est insérée dans la docstring : c'est ce que lit le LLM pour choisir la statistique.
rang_joueur.__doc__ = (rang_joueur.__doc__ or "").replace("{liste}", "\n".join("            " + l for l in LISTE_STATS.splitlines()))
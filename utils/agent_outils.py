# utils/agent_outils.py — dépendances et outil SQL de l'agent Pydantic AI (étape 2, option A).
#
# La recherche vectorielle reste systématique (comme en v1) ; l'agent dispose EN PLUS de l'outil
# interroger_statistiques, qu'il appelle pour les questions chiffrées.
# Chaque appel de l'outil complète l'objet ContexteRecherche partagé avec l'agent :
#   - contextes          : le résultat SQL en texte -> évalué par RAGAS (faithfulness) comme un document ;
#   - sources_autorisees : SOURCE_SQL ajoutée -> le validateur de ReponseAssistant accepte de la citer ;
#   - requetes_sql       : trace de chaque requête (pour reponses.json et l'analyse des erreurs).
import threading
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from sql_tool import interroger_base

SOURCE_SQL = "Base de données NBA (SQL)"
MAX_LIGNES_TEXTE = 20  # lignes du résultat transmises au LLM (le reste est signalé, pas envoyé)
MAX_APPELS_SQL = 2     # appels de l'outil par réponse : au-delà, l'agent « pêche » des chiffres hors sujet


@dataclass
class ContexteRecherche:
    """Objet transmis à l'agent (deps) et modifié par les outils pendant la réponse."""

    sources_autorisees: list[str]
    contextes: list[str]
    requetes_sql: list[dict] = field(default_factory=list)
    nb_appels: int = 0  # appels d'outils de données déjà RÉSERVÉS (voir reserver_appel)
    verrou: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def reserver_appel(ctx: RunContext[ContexteRecherche]) -> bool:
    """Réserve un appel d'outil de données ; False si la limite MAX_APPELS_SQL est atteinte.

    Pydantic AI exécute les appels d'outils PARALLÈLES dans des threads : « lire le compteur, puis
    l'incrémenter plus tard » laisse passer plusieurs appels à la fois (40 appels observés en v2d).
    Le verrou rend « vérifier + incrémenter » indivisible."""
    with ctx.deps.verrou:
        if ctx.deps.nb_appels >= MAX_APPELS_SQL:
            return False
        ctx.deps.nb_appels += 1
        return True


def resultat_en_texte(resultat: dict) -> str:
    """Résultat de interroger_base() -> texte lisible par le LLM et par le juge RAGAS."""
    entete = f"[Source : {SOURCE_SQL}] Question : {resultat['question']}\nRequête : {resultat['requete']}\n"
    if "erreur" in resultat:
        return entete + f"Erreur : {resultat['erreur']}. Aucune donnée obtenue."
    if not resultat["lignes"]:
        return entete + "Résultat : aucune ligne. La base ne contient pas cette information."
    lignes = [
        " | ".join(f"{col} = {val}" for col, val in zip(resultat["colonnes"], ligne))
        for ligne in resultat["lignes"][:MAX_LIGNES_TEXTE]
    ]
    texte = entete + "Résultat :\n" + "\n".join(f"- {l}" for l in lignes)
    if resultat["tronque"] or len(resultat["lignes"]) > MAX_LIGNES_TEXTE:
        texte += f"\n(résultat tronqué aux {MAX_LIGNES_TEXTE} premières lignes)"
    return texte


def interroger_statistiques(ctx: RunContext[ContexteRecherche], question: str) -> str:
    """Interroge la base de statistiques NBA de la saison régulière 2024-25 complète (une ligne par
    joueur : totaux de saison, moyennes par match sur la saison, pourcentages, ratings). À utiliser pour
    toute question chiffrée sur la saison : valeur d'un joueur, classement, comparaison, filtre sur un
    seuil, statistiques d'une équipe. La base ne contient AUCUNE donnée par match ni par période
    (derniers matchs, un mois, un match précis) : pour ces questions, ne pas l'utiliser.

    Args:
        question: la question statistique précise, en français, avec les noms et seuils utiles.
    """
    if not reserver_appel(ctx):
        return (f"Limite de {MAX_APPELS_SQL} interrogations atteinte. Réponds avec les résultats déjà "
                "obtenus, ou indique que l'information n'est pas disponible.")
    resultat = interroger_base(question)
    ctx.deps.requetes_sql.append({
        "question": question,
        "requete": resultat["requete"],
        "tentatives": resultat["tentatives"],
        "nb_lignes": len(resultat.get("lignes", [])),
        "erreur": resultat.get("erreur"),
    })
    texte = resultat_en_texte(resultat)
    if "erreur" not in resultat:  # une erreur n'est pas une donnée : ni contexte, ni source citable
        ctx.deps.contextes.append(texte)
        if SOURCE_SQL not in ctx.deps.sources_autorisees:
            ctx.deps.sources_autorisees.append(SOURCE_SQL)
    return texte
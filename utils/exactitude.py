# utils/exactitude.py — métrique déterministe d'exactitude pour les questions chiffrées ou nominatives.
#
# Pourquoi : la vérification manuelle de la v1b a montré que answer_correctness (RAGAS) ne distingue pas
# une réponse juste d'une réponse fausse sur un chiffre (S10 : « 9,8 » au lieu de 8,5 noté 0,75, comme
# des réponses justes). Ici, pas de juge LLM : on vérifie que les éléments attendus (nombres, noms)
# figurent bien dans la réponse.
#
# Un « élément attendu » est :
#   - un nombre (int ou float)            -> présent dans la réponse à TOLERANCE près ;
#   - un texte (nom de joueur, d'équipe)  -> présent comme mot entier, sans tenir compte casse/accents ;
#   - une liste d'alternatives            -> au moins une alternative présente (ex. ["MIN", "Minnesota Timberwolves"]).
# Score d'une réponse = part des éléments attendus trouvés (1.0 = réponse exacte).
import re
import unicodedata

TOLERANCE = 0.05  # 8,5 attendu : 8,5 ou 8,46 acceptés ; 9,8 ou 8,6 refusés

# Nombres : « 1 920 » (espace, espace insécable ou fine), « 33,2 », « 8.5 », « 2485 »
_MOTIF_NOMBRE = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


def extraire_nombres(texte: str) -> list[float]:
    """Tous les nombres d'un texte, au format float (séparateurs de milliers et virgule décimale gérés)."""
    nombres = []
    for brut in _MOTIF_NOMBRE.findall(texte):
        propre = re.sub(r"[   ]", "", brut).replace(",", ".")
        nombres.append(float(propre))
    return nombres


def normaliser(texte: str) -> str:
    """Minuscules, sans accents (Jokić -> jokic), tirets et ponctuation remplacés par des espaces."""
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c)).lower()
    texte = re.sub(r"[^a-z0-9]+", " ", texte)
    return f" {texte.strip()} "  # espaces en bord : permet de chercher un mot entier avec « f' {mot} ' »


def element_present(reponse: str, element) -> bool:
    if isinstance(element, list):  # alternatives
        return any(element_present(reponse, alt) for alt in element)
    if isinstance(element, (int, float)):
        return any(abs(n - element) <= TOLERANCE for n in extraire_nombres(reponse))
    return normaliser(element) in normaliser(reponse)


def score_exactitude(reponse: str, elements_attendus: list) -> float | None:
    """Part des éléments attendus présents dans la réponse ; None si la question n'en a pas."""
    if not elements_attendus:
        return None
    trouves = sum(element_present(reponse, e) for e in elements_attendus)
    return trouves / len(elements_attendus)
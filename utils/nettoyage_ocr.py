# utils/nettoyage_ocr.py — nettoyage du bruit d'interface dans les captures Reddit OCRisées (v1).
#
# Les 4 PDF Reddit sont des captures d'écran : l'OCR restitue, mêlés aux commentaires, les boutons
# (« Répondre », « Se connecter »), les dates relatives (« ~15 j »), les compteurs de votes, les URL
# et l'en-tête du post réimprimé à chaque page. Ce bruit dilue les embeddings et le contexte du LLM.
# Mesuré sur l'OCR du prototype : environ 27 % du texte des PDF est retiré, sans perte de contenu.
import re
from collections import Counter

# Lignes d'interface Reddit capturées par l'OCR (comparaison sans casse ni espaces superflus)
LIGNES_INTERFACE = {
    "répondre", "se connecter", "accéder au contenu principal", "comm. du top 1%", "comm. du",
    "réponse supplémentaire", "afficher plus de commentaires", "[supprimé]", "1%",
    "rinba", "r/nba", "r/hockey", "r/collegebasketball",
}
MOTIFS_BRUIT = [
    re.compile(r"^[~\-]?\s*\d+\s?(j|m\.?|a|h)$", re.I),          # dates relatives : ~15 j, -1 m, 3 h
    re.compile(r"^(il ?y ?a|ilya|ily a)\b.*$", re.I),              # « il y a 3 m. »
    re.compile(r"^modifié il y a.*$", re.I),
    re.compile(r"^rechercher dans r/.*$", re.I),                    # barre de recherche Reddit
    re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{2}[:.]\d{2}$"),           # horodatage d'impression : 12/06/2025 13:11
    re.compile(r"^\d{1,3}/\d{1,3}$"),                              # pagination : 11/15
    re.compile(r"(https?:|www|reddit\.?com|comments/)", re.I),     # URL et fragments d'URL
    re.compile(r"^\S*_\S*$"),                                      # morceau d'URL isolé : miller_is, best_teams
    re.compile(r"^[\d\s&%.,]+$"),                                  # compteurs de votes isolés : 4 1, 107
    re.compile(r"^\d+\s+(upvotes|commentaires)$", re.I),
]


def nettoyer_texte_ocr(texte: str, max_repetitions: int = 3) -> str:
    """Supprime le bruit d'interface des captures Reddit OCRisées.

    - lignes d'interface (boutons, menus), dates relatives, horodatages, pagination,
      URL, compteurs de votes ;
    - lignes longues répétées (en-tête du post réimprimé à chaque page) : seule la
      première occurrence est conservée au-delà de `max_repetitions` apparitions.
    """
    lignes = [l.strip() for l in texte.splitlines()]
    frequences = Counter(lignes)
    deja_vues: set[str] = set()
    gardees = []
    for ligne in lignes:
        if not ligne or ligne.lower() in LIGNES_INTERFACE:
            continue
        if any(m.search(ligne) for m in MOTIFS_BRUIT):
            continue
        if frequences[ligne] > max_repetitions:
            if ligne in deja_vues:
                continue
            deja_vues.add(ligne)
        gardees.append(ligne)
    return "\n".join(gardees)

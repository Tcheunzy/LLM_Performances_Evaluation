# setup_rag.py — met en place le système RAG complet en un seul lancement.
#
# Étapes (chacune est une fonction, journalisée avec sa durée) :
#   1. verifier_environnement : clé API, fichiers d'entrée, dossiers ;
#   2. construire_index       : index vectoriel FAISS (sauté s'il existe déjà, sauf --force-index) ;
#   3. construire_base        : base SQLite validée par Pydantic (load_excel_to_db.charger_base) ;
#   4. tester_systeme         : deux questions de contrôle posées au pipeline complet.
# Le journal est affiché ET écrit dans logs/setup_rag.log ; le script s'arrête à la première étape en échec.
# Lancement : uv run setup_rag.py                  (tout)
#             uv run setup_rag.py --force-index    (reconstruire aussi l'index, ~ plusieurs minutes d'OCR)
#             uv run setup_rag.py --etapes base test
import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "setup_rag.log"
ETAPES = ["environnement", "index", "base", "test"]

logger = logging.getLogger("setup_rag")


def configurer_logging() -> None:
    """Journal structuré : horodatage | niveau | composant | message, en console et dans un fichier."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    format_ = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    racine = logging.getLogger()
    racine.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(format_)
        racine.addHandler(handler)


@contextmanager
def etape(nom: str, bilan: list):
    """Journalise le début, la fin, la durée et l'éventuelle erreur d'une étape."""
    logger.info("=== Étape « %s » : début ===", nom)
    debut = time.perf_counter()
    try:
        yield
    except Exception:
        duree = time.perf_counter() - debut
        logger.exception("Étape « %s » en échec après %.1f s", nom, duree)
        bilan.append((nom, "échec", duree))
        raise
    duree = time.perf_counter() - debut
    logger.info("=== Étape « %s » : terminée en %.1f s ===", nom, duree)
    bilan.append((nom, "ok", duree))


# --------------------------------------------------------------------------- étapes
def verifier_environnement() -> None:
    """Vérifie ce dont les autres étapes ont besoin, AVANT de lancer quoi que ce soit de long."""
    load_dotenv(ROOT / ".env")
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("MISTRAL_API_KEY absente : copier .env.example en .env et y mettre la clé.")
    from utils.config import INPUT_DIR, VECTOR_DB_DIR

    dossier_entree = ROOT / INPUT_DIR
    excel = dossier_entree / "regular NBA.xlsx"
    if not excel.exists():
        raise FileNotFoundError(f"Fichier de statistiques introuvable : {excel}")
    pdfs = sorted(dossier_entree.glob("*.pdf"))
    logger.info("Entrées : %s + %d PDF dans %s", excel.name, len(pdfs), dossier_entree)
    logger.info("Index vectoriel : %s", ROOT / VECTOR_DB_DIR)


def construire_index(force: bool = False) -> None:
    """Index FAISS : réutilisé s'il existe (l'OCR des PDF est long), reconstruit si force=True."""
    from utils.config import INPUT_DIR, VECTOR_DB_DIR

    dossier = ROOT / VECTOR_DB_DIR
    if not force and (dossier / "faiss_index.idx").exists() and (dossier / "document_chunks.pkl").exists():
        logger.info("Index existant réutilisé (%s). Option --force-index pour le reconstruire.", dossier)
        return
    from indexer import run_indexing

    logger.info("Construction de l'index dans %s (OCR + embeddings : plusieurs minutes)", dossier)
    run_indexing(input_directory=str(ROOT / INPUT_DIR))
    if not (dossier / "faiss_index.idx").exists():
        raise RuntimeError(f"L'indexation n'a produit aucun index dans {dossier}.")


def construire_base() -> dict:
    """Base SQLite reconstruite à partir de l'Excel, ligne par ligne validée par Pydantic."""
    from load_excel_to_db import charger_base

    resume = charger_base()
    logger.info("Base prête : %(equipes)d équipes, %(joueurs)d joueurs, %(rejeté)d rejet(s)", resume)
    return resume


QUESTIONS_DE_CONTROLE = [
    ("Quel est le pourcentage de réussite à 3 points de James Harden ?", True),   # doit répondre
    ("Qui a gagné le tournoi de Roland-Garros en 2025 ?", False),               # doit refuser
]


def tester_systeme() -> None:
    """Pose deux questions au pipeline complet : une à laquelle il doit répondre, une qu'il doit refuser."""
    from rag_pipeline import MESSAGE_ERREUR, answer  # importé ici : charge l'index, qui doit exister

    for question, doit_repondre in QUESTIONS_DE_CONTROLE:
        debut = time.perf_counter()
        resultat = answer(question)
        duree = time.perf_counter() - debut
        if resultat["answer"] == MESSAGE_ERREUR:
            raise RuntimeError(f"Erreur technique sur la question de contrôle : {question}")
        conforme = resultat.get("information_disponible") == doit_repondre
        logger.log(logging.INFO if conforme else logging.WARNING,
                   "Contrôle %s en %.1f s | attendu : %s | obtenu : information_disponible=%s | %s",
                   "OK" if conforme else "NON CONFORME", duree, "réponse" if doit_repondre else "refus",
                   resultat.get("information_disponible"), resultat["answer"][:100])


# --------------------------------------------------------------------------- orchestration
def main(etapes: list[str], force_index: bool) -> int:
    bilan: list[tuple[str, str, float]] = []
    actions = {
        "environnement": verifier_environnement,
        "index": lambda: construire_index(force=force_index),
        "base": construire_base,
        "test": tester_systeme,
    }
    try:
        for nom in ETAPES:  # toujours dans l'ordre, même si l'utilisateur les cite dans un autre
            if nom in etapes:
                with etape(nom, bilan):
                    actions[nom]()
    except Exception:
        code = 1
    else:
        code = 0
    logger.info("--- Bilan ---")
    for nom, statut, duree in bilan:
        logger.info("%-14s %-6s %6.1f s", nom, statut, duree)
    logger.info("Système %s. Journal complet : %s", "opérationnel" if code == 0 else "NON opérationnel", LOG_FILE)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mise en place du système RAG SportSee en un seul lancement")
    parser.add_argument("--etapes", nargs="+", choices=ETAPES, default=ETAPES,
                        help="étapes à exécuter (par défaut : toutes, dans l'ordre)")
    parser.add_argument("--force-index", action="store_true", help="reconstruire l'index FAISS même s'il existe")
    args = parser.parse_args()
    configurer_logging()
    sys.exit(main(args.etapes, args.force_index))
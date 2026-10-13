# api.py — API REST de l'assistant SportSee (FastAPI).
#
# Routes :
#   POST /ask    : pose une question à l'assistant (pipeline RAG + outil SQL) ;
#   GET  /health : état du service (index chargé, base accessible) ;
#   GET  /docs   : documentation interactive générée automatiquement (Swagger UI).
# Les entrées et sorties sont décrites par des modèles Pydantic : FastAPI valide la requête (422 si invalide)
# et documente le format de la réponse.
# Lancement : uv run uvicorn api:app --reload      puis http://127.0.0.1:8000/docs
import logging
import sqlite3
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from load_excel_to_db import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("api")


# --------------------------------------------------------------------------- schémas
class Question(BaseModel):
    """Corps de la requête POST /ask."""
    question: str = Field(min_length=3, max_length=500,
                          examples=["Quel est le pourcentage de réussite à 3 points de James Harden ?"])

    @field_validator("question")
    @classmethod
    def non_vide(cls, valeur: str) -> str:
        valeur = valeur.strip()
        if len(valeur) < 3:
            raise ValueError("la question est vide ou trop courte")
        return valeur


class RequeteSQLExecutee(BaseModel):
    """Trace d'une interrogation de la base par l'outil SQL."""
    question: str
    requete: str | None
    nb_lignes: int
    erreur: str | None = None


class Reponse(BaseModel):
    """Corps de la réponse POST /ask."""
    question: str
    reponse: str
    information_disponible: bool | None = Field(
        description="False si l'assistant indique que l'information n'est pas dans ses données")
    sources: list[str] = Field(description="sources citées par l'assistant (documents ou base SQL)")
    requetes_sql: list[RequeteSQLExecutee]
    duree_s: float


class Sante(BaseModel):
    """Corps de la réponse GET /health."""
    statut: str
    index_vectoriel: int = Field(description="nombre d'extraits dans l'index FAISS")
    joueurs_en_base: int
    version: str


# --------------------------------------------------------------------------- application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : charge le pipeline UNE fois (index FAISS, agent), pas à chaque question."""
    import rag_pipeline

    app.state.pipeline = rag_pipeline
    logger.info("Pipeline chargé (%s)", rag_pipeline.PIPELINE_DESCRIPTION[:60])
    yield

class JSONUTF8(JSONResponse):
    """Déclare l'encodage dans l'en-tête (certains clients, dont PowerShell 5.1, ne supposent pas l'UTF-8)."""
    media_type = "application/json; charset=utf-8"

app = FastAPI(
    title="SportSee — Assistant NBA",
    description="Assistant RAG pour les coachs : documents (recherche FAISS) et statistiques "
                "de la saison régulière 2024-25 (base SQL).",
    version="2b",
    lifespan=lifespan,
    default_response_class=JSONUTF8,
)


@app.post("/ask", response_model=Reponse)
def ask(corps: Question, request: Request) -> Reponse:
    # « def » et non « async def » : answer() est bloquant (appels HTTP à Mistral) ;
    # FastAPI l'exécute alors dans un thread, sans bloquer les autres requêtes.
    pipeline = request.app.state.pipeline
    debut = time.perf_counter()
    resultat = pipeline.answer(corps.question)
    duree = time.perf_counter() - debut

    if resultat["answer"] == pipeline.MESSAGE_ERREUR:
        logger.error("Échec du pipeline en %.1f s | %s", duree, corps.question)
        raise HTTPException(status_code=503, detail="L'assistant n'a pas pu produire de réponse. Réessayez.")

    logger.info("Réponse en %.1f s | information_disponible=%s | %d requête(s) SQL | %s", duree,
                resultat.get("information_disponible"), len(resultat.get("requetes_sql", [])), corps.question)
    return Reponse(
        question=corps.question,
        reponse=resultat["answer"],
        information_disponible=resultat.get("information_disponible"),
        sources=resultat.get("sources_citees", []),
        requetes_sql=resultat.get("requetes_sql", []),
        duree_s=round(duree, 2),
    )


@app.get("/health", response_model=Sante)
def health(request: Request) -> Sante:
    pipeline = request.app.state.pipeline
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as connexion:
            joueurs = connexion.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        index = pipeline.vector_store_manager.index.ntotal
    except Exception as erreur:
        logger.exception("Health check en échec")
        raise HTTPException(status_code=503, detail=f"Service indisponible : {erreur}")
    return Sante(statut="ok", index_vectoriel=index, joueurs_en_base=joueurs, version=app.version)
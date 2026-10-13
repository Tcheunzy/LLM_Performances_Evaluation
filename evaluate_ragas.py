import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from rich.progress import track

warnings.filterwarnings("ignore", category=DeprecationWarning)
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

from bilan_seuils import calculer_bilan
from rag_pipeline import MESSAGE_ERREUR, PIPELINE_DESCRIPTION, answer
from utils.config import EMBEDDING_MODEL, JUDGE_MODEL, MISTRAL_API_KEY, MODEL_NAME, VECTOR_DB_DIR

#Instanciation du module argparse pour le versionning des évaluations en fonction de ce qu'on évalue (notion de version du prototype):
parser = argparse.ArgumentParser(description="Evaluation RAGAS de l'assistant NBA")
parser.add_argument("--version", required=True, help="Nom de la version de prototype évalué par Ragas (ex:V0_baseline) : dossier de resultats")
parser.add_argument("--limit", type=int, default=None, help="Nombre de questions à évaluer (par défaut : toutes)")
args = parser.parse_args()


#Création des chemins vers données et d'output:
ROOT = Path(__file__).resolve().parent
TESTSET = ROOT / "eval" / "testset.json"
OUTPUT_DIR = ROOT / "eval" / "results" / args.version
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Journalisation : console + fichier evaluation.log dans le dossier de la version (trace conservée avec les résultats).
#Niveau WARNING par défaut (pipeline, httpx, ragas restent discrets), INFO pour ce script.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / "evaluation.log", mode="w", encoding="utf-8")],
    force=True,
)
logger = logging.getLogger("evaluate_ragas")
logger.setLevel(logging.INFO)

#Chargement du jeu de questions tests:
df = pd.read_json(TESTSET)
if args.limit:
    df = df.head(args.limit)
logger.info("Version %s | %d questions chargées | pipeline : %s", args.version, len(df), PIPELINE_DESCRIPTION)

#Création des listes vides acceuillant les réponses et le contextes
responses = []
contextes = []
disponibles = []
requetes_sql = []
durees = []

debut_generation = time.perf_counter()
for q in track(df["question"], description="Traitement des données"):
    debut = time.perf_counter()
    result = answer(q)
    durees.append(round(time.perf_counter() - debut, 2))
    responses.append(result["answer"])
    contextes.append(result["contexts"])
    disponibles.append(result.get("information_disponible"))   # None pour les versions sans agent
    requetes_sql.append(result.get("requetes_sql", []))

df["answer"] = responses
df["contexts"] = contextes
df["information_disponible"] = disponibles
df["requetes_sql"] = requetes_sql
df["duree_s"] = durees
logger.info("Génération des réponses : %.0f s (médiane %.1f s / question, max %.1f s)",
            time.perf_counter() - debut_generation, df["duree_s"].median(), df["duree_s"].max())

#Gestion d'erreur : le message est celui que renvoie answer() en cas d'échec technique
df["erreur"] = df["answer"] == MESSAGE_ERREUR
for _, ligne in df[df["erreur"]].iterrows():
    logger.warning("Réponse en erreur : %s | %s", ligne["id"], ligne["question"])
logger.info("Réponses en erreur : %d / %d", df["erreur"].sum(), len(df))

file_path = OUTPUT_DIR / "reponses.json"
df.to_json(file_path, orient="records", indent=2, force_ascii=False)

# Taux de refus correct : questions hors données / hors sujet pour lesquelles l'assistant a signalé l'absence d'information
refus = df[df["category"].isin(["hors_donnees", "hors_sujet"])]
if refus["information_disponible"].notna().any():
    nb_refus = (refus["information_disponible"] == False).sum()  # noqa: E712
    logger.info("Taux de refus correct : %d / %d", nb_refus, len(refus))

#Sauvegarde de la configuration de l'évaluation (traçabilité)
config_eval = {
    "version": args.version,
    "date": datetime.now().isoformat(timespec="seconds"),
    "modele_evalue": MODEL_NAME,
    "modele_juge": JUDGE_MODEL,
    "modele_embeddings": EMBEDDING_MODEL,
    "nb_questions": len(df),
    "index_vectoriel": VECTOR_DB_DIR,
    "pipeline": PIPELINE_DESCRIPTION,
}
with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config_eval, f, ensure_ascii=False, indent=2)


# Construction du dataset RAGAS
# On ne note que les réponses valides (les messages d'erreur sont exclus et comptés à part).
df_valide = df[~df["erreur"]].reset_index(drop=True)

echantillons = [
    {
        "user_input": ligne["question"],          # la question posée
        "response": ligne["answer"],              # la réponse de l'assistant (ministral-3b)
        "retrieved_contexts": ligne["contexts"],  # la liste des textes récupérés par le retriever
        "reference": ligne["ground_truth"],       # la bonne réponse, issue du jeu de test
    }
    for _, ligne in df_valide.iterrows()
]

dataset = EvaluationDataset.from_list(echantillons)
logger.info("Dataset RAGAS : %d questions à noter", len(dataset))

#Ajout du Juge pour pallier au problème de l'answer relevancy
class ChatMistralAIJuge(ChatMistralAI):
    """ChatMistralAI corrigé : langchain-mistralai tente d'additionner les compteurs de tokens,
    or l'API Mistral en renvoie certains sous forme de dictionnaire (ex. prompt_tokens_details),
    ce qui fait planter answer_relevancy (qui combine plusieurs générations).
    On n'additionne ici que les compteurs numériques."""

    def _combine_llm_outputs(self, llm_outputs):
        total = {}
        for out in llm_outputs:
            if not out or not out.get("token_usage"):
                continue
            for cle, valeur in out["token_usage"].items():
                if isinstance(valeur, (int, float)):
                    total[cle] = total.get(cle, 0) + valeur
        return {"token_usage": total, "model_name": self.model}

#Instanciation des juges (llm+embeddings)
juge_llm = LangchainLLMWrapper(
    ChatMistralAIJuge(model=JUDGE_MODEL, mistral_api_key=MISTRAL_API_KEY, temperature=0)
)
juge_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(model=EMBEDDING_MODEL, mistral_api_key=MISTRAL_API_KEY)
)

#Runconfig ragas
run_config = RunConfig(timeout=120, max_retries=6, max_workers=1, max_wait=60, seed=42)

#Evaluation Ragas
debut_ragas = time.perf_counter()
ragas_result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
    llm=juge_llm,
    embeddings=juge_embeddings,
    run_config=run_config,
)
logger.info("Notation RAGAS : %.0f s", time.perf_counter() - debut_ragas)

#Scores par question:
METRIQUES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
scores = ragas_result.to_pandas()
scores.insert(0, "id", df_valide["id"])               # même ordre que df_valide (grâce au reset_index)
scores.insert(1, "category", df_valide["category"])
scores.to_csv(OUTPUT_DIR / "scores_par_question.csv", index=False, encoding="utf-8-sig")

#Une métrique non calculée (NaN) signale souvent un échec du juge (timeout, format) : on la journalise
manquants = scores[METRIQUES].isna().sum()
for metrique, n in manquants[manquants > 0].items():
    logger.warning("%s non calculée pour %d question(s)", metrique, n)

#Scores moyens par catégorie
par_categorie = scores.groupby("category")[METRIQUES].mean().round(3)
par_categorie.loc["GLOBAL"] = scores[METRIQUES].mean().round(3)
par_categorie.to_csv(OUTPUT_DIR / "scores_par_categorie.csv", encoding="utf-8-sig")
logger.info("Scores moyens par catégorie :\n%s", par_categorie.to_string())

bilan = calculer_bilan(df, scores)
bilan.to_csv(OUTPUT_DIR / "bilan_seuils.csv", index=False, encoding="utf-8-sig")
logger.info("Bilan par rapport aux seuils (min = bêta, cible = production) :\n%s", bilan.to_string(index=False))
logger.info("Résultats enregistrés dans %s", OUTPUT_DIR)
import argparse
import json
from pathlib import Path
from rich.progress import track
from datetime import datetime
import logging
logging.getLogger().setLevel(logging.WARNING)

import pandas as pd
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness, answer_correctness

from rag_pipeline import answer
from utils.config import (
    MISTRAL_API_KEY,
    EMBEDDING_MODEL,
    MODEL_NAME,
    JUDGE_MODEL,
    VECTOR_DB_DIR,
)
from rag_pipeline import PIPELINE_DESCRIPTION, answer
from bilan_seuils import calculer_bilan

#Instanciation du module argparse pour le versionning des évaluations en fonction de ce qu'on évalue (notion de version du prototype):
parser = argparse.ArgumentParser(description="Evaluation RAGAS de l'assistant NBA")
parser.add_argument("--version", required=True, help="Nom de la version de prototype évalué par Ragas (ex:V0_baseline) : dossier de resultats")
parser.add_argument("--limit", type=int, default=None, help="Nombre de questions à évaluer (par défaut : toutes)")
args = parser.parse_args()


#Création des chemins vers données et d'output:
ROOT = Path(__file__).resolve().parent
TESTSET=ROOT / "eval" / "testset.json"
OUTPUT_DIR =ROOT / "eval" / "results" / args.version
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Chargement du jeu de questions tests:
df = pd.read_json(TESTSET)
if args.limit:
    df = df.head(args.limit)      
print(f"Nombre de questions chargées : {len(df)}")

#Instanciation des scorers RAGAS:
juge_llm = LangchainLLMWrapper(
    ChatMistralAI(model=JUDGE_MODEL, mistral_api_key=MISTRAL_API_KEY, temperature=0)
)
juge_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(model=EMBEDDING_MODEL, mistral_api_key=MISTRAL_API_KEY)
)

#Création des listes vides acceuillant les réponses et le contextes
responses=[]
contextes=[]
disponibles=[]

for q in track(df["question"], description="Traitement des données"):
    result = answer(q)
    responses.append(result["answer"])
    contextes.append(result["contexts"])
    disponibles.append(result.get("information_disponible"))   # None pour les versions sans agent

df["answer"] = responses
df["contexts"] = contextes
df["information_disponible"] = disponibles

#Ajout gestion d'erreur
MESSAGE_ERREUR = "Je suis désolé, une erreur technique"
df["erreur"] = df["answer"].str.startswith(MESSAGE_ERREUR)
print(f"Réponses en erreur : {df['erreur'].sum()} / {len(df)}")

file_path = OUTPUT_DIR / "reponses.json"
df.to_json(file_path, orient="records", indent=2, force_ascii=False)

# Taux de refus correct : questions hors données / hors sujet pour lesquelles l'assistant a signalé l'absence d'information
refus = df[df["category"].isin(["hors_donnees", "hors_sujet"])]
if refus["information_disponible"].notna().any():
    nb_refus = (refus["information_disponible"] == False).sum()  # noqa: E712
    print(f"Taux de refus correct : {nb_refus} / {len(refus)}")

#Sauvegarde de la configuration de l'évaluation (traçabilité)
config_eval = {
    "version": args.version,
    "date": datetime.now().isoformat(timespec="seconds"),
    "modele_evalue": MODEL_NAME,
    "modele_juge": JUDGE_MODEL,
    "modele_embeddings": EMBEDDING_MODEL,
    "nb_questions": len(df),
    "index_vectoriel": VECTOR_DB_DIR,
    "prompt": "SYSTEM_PROMPT_V1 + agent Pydantic AI",
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
print(f"Dataset RAGAS : {len(dataset)} questions à noter")

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
ragas_result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
    llm=juge_llm,
    embeddings=juge_embeddings,
    run_config=run_config,
)

#Scores par question:
METRIQUES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
scores = ragas_result.to_pandas()
scores.insert(0, "id", df_valide["id"])               # même ordre que df_valide (grâce au reset_index)
scores.insert(1, "category", df_valide["category"])
scores.to_csv(OUTPUT_DIR / "scores_par_question.csv", index=False, encoding="utf-8-sig")

#Scores moyens par catégorie
par_categorie = scores.groupby("category")[METRIQUES].mean().round(3)
par_categorie.loc["GLOBAL"] = scores[METRIQUES].mean().round(3)
par_categorie.to_csv(OUTPUT_DIR / "scores_par_categorie.csv", encoding="utf-8-sig")

print("\n=== Scores moyens par catégorie ===")
print(par_categorie.to_string())
print(f"\nRésultats enregistrés dans {OUTPUT_DIR}")

bilan = calculer_bilan(df, scores)
bilan.to_csv(OUTPUT_DIR / "bilan_seuils.csv", index=False, encoding="utf-8-sig")
print("\n=== Bilan par rapport aux seuils (min = bêta, cible = production) ===")
print(bilan.to_string(index=False))
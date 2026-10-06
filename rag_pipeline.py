#Fichier contenant toute la logique RAG du prototype afin d'effectuer proprement l'évaluation RAGAS. 
import logging
from utils.config import MISTRAL_API_KEY,MODEL_NAME,SEARCH_K, MISTRAL_RETRY_CONFIG
from utils.vector_store import VectorStoreManager
from mistralai.client import Mistral

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from utils.schemas import ReponseAssistant
import logfire

# Description du pipeline, enregistrée dans config.json à chaque évaluation (traçabilité)
PIPELINE_DESCRIPTION = "v1b : SYSTEM_PROMPT_V1 (coachs) + agent Pydantic AI (sortie ReponseAssistant)"

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

# --- Observabilité : Logfire ---
# send_to_logfire="if-token-present" : les traces partent vers Logfire si un projet est configuré,
# sinon le pipeline fonctionne normalement sans rien envoyer (utile pour un clone du dépôt).
logfire.configure(service_name="sportsee-assistant", send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()   # trace automatiquement chaque appel de l'agent (prompt, réponse, tokens, durée)

#On instancie d'abord le client mistral.
def get_mistral_client(api_key: str) -> Mistral:
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY absente : définis-la dans le fichier .env.")
    client = Mistral(api_key=api_key, retry_config=MISTRAL_RETRY_CONFIG)
    logging.info("Client Mistral initialisé.")
    return client

client = get_mistral_client(MISTRAL_API_KEY)


#On charge le vectorstore qui contient toute la logique de récupération du texte et son chargement.
def get_vector_store_manager() -> VectorStoreManager:
    logging.info("Chargement du VectorStoreManager...")
    manager = VectorStoreManager()
    if manager.index is None or not manager.document_chunks:
        raise RuntimeError("Index FAISS ou chunks introuvables dans vector_db/. Lance indexer.py.")
    if manager.index.ntotal != len(manager.document_chunks):
        raise RuntimeError(
            f"Index incohérent : {manager.index.ntotal} vecteurs pour {len(manager.document_chunks)} chunks."
        )
    logging.info(f"VectorStoreManager chargé ({manager.index.ntotal} vecteurs).")
    return manager

vector_store_manager = get_vector_store_manager()

#Prompt système.
# --- Prompt RAG (prototype d'origine, conservé tel quel pour la baseline) ---
# Chaîne classique (pas une f-string) : {context_str} et {question} sont remplis avec .format()
SYSTEM_PROMPT_V0 = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

SYSTEM_PROMPT_V1 = """Tu es l'assistant d'analyse de performance de SportSee. Tu aides les entraîneurs, \
analystes et préparateurs physiques d'un club de basketball à retrouver rapidement des informations fiables \
sur la saison régulière NBA.

RÈGLES IMPÉRATIVES
1. Réponds UNIQUEMENT à partir des documents fournis dans le contexte. N'utilise jamais tes connaissances \
générales, même si tu penses connaître la réponse.
2. Si le contexte ne contient pas l'information demandée, dis-le explicitement \
(« Cette information n'est pas disponible dans les données. »), sans proposer d'estimation.
3. Si la question ne concerne pas la NBA, indique qu'elle sort de ton périmètre, sans y répondre.
4. Reprends les chiffres exactement tels qu'ils figurent dans le contexte. Distingue les totaux sur la \
saison et les moyennes par match, comme l'indiquent les libellés. Ne fais un calcul que s'il est simple, \
et précise alors comment tu l'as obtenu.
5. Les données couvrent une seule saison régulière, en statistiques cumulées par joueur : elles ne \
contiennent ni résultats match par match, ni distinction domicile / extérieur, ni playoffs.
6. Les extraits Reddit sont des opinions de fans : présente-les comme telles (« selon des commentaires \
Reddit… »), jamais comme des faits.

FORMAT
- Réponse directe en premier, en une à trois phrases. Ajoute des détails seulement s'ils sont utiles.
- Cite la ou les sources utilisées entre crochets, par exemple [regular NBA.xlsx (Joueur: James Harden)].
- Ton professionnel et factuel, en français, sans émojis."""

USER_PROMPT_V1 = """CONTEXTE :
---
{context_str}
---

QUESTION :
{question}"""



# Agent Pydantic AI : même client Mistral (avec relances), sortie validée par ReponseAssistant
modele_llm = MistralModel(MODEL_NAME, provider=MistralProvider(mistral_client=client))

agent = Agent(
    modele_llm,
    output_type=ReponseAssistant,      # le LLM DOIT renvoyer un objet conforme à ce modèle
    instructions=SYSTEM_PROMPT_V1,     # le prompt système orienté coachs
    retries=2,                         # si la sortie est invalide, l'agent redemande (2 fois max)
    model_settings={"temperature": 0.1},
)


#Formatage du contexte donné par le retriever.
def formater_contexte(search_results: list[dict]) -> str:
    """Transforme les résultats de la recherche vectorielle en un bloc de texte pour le prompt."""
    if not search_results:
        return "Aucune information pertinente trouvée dans la base de connaissances pour cette question."

    return "\n\n---\n\n".join(
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\n"
        f"Contenu: {res['text']}"
        for res in search_results
    )

#Gestion de l'envoie du prompt augmenté et récupération de la réponse du LLM
model = MODEL_NAME
def generer_reponse(prompt_messages: list[dict]) -> str:
    """Envoie les messages (contexte inclus) à l'API Mistral et renvoie le texte de la réponse."""
    if not prompt_messages:
        logging.warning("Tentative de génération de réponse avec un prompt vide.")
        return "Je ne peux pas traiter une demande vide."

    try:
        logging.info(f"Appel à l'API Mistral, modèle '{model}', avec {len(prompt_messages)} message(s).")
        response = client.chat.complete(
            model=model,
            messages=prompt_messages,
            temperature=0.1,  # Température basse pour des réponses factuelles
        )

        if response.choices:
            logging.info("Réponse reçue de l'API Mistral.")
            return response.choices[0].message.content

        logging.warning("L'API n'a pas retourné de choix valide.")
        return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."

    except Exception as e:
        logging.exception("Erreur API Mistral pendant client.chat.complete")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."


#Ajout de la fonction answer() qui sert à l'orchestration du pipeline.
MESSAGE_ERREUR = "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."


def answer(question: str) -> dict:
    with logfire.span("answer", question=question):

        # 1. Retrieval
        with logfire.span("retrieval", k=SEARCH_K) as span:
            search_results = vector_store_manager.search(question, k=SEARCH_K)
            span.set_attribute("sources", [r["metadata"].get("source") for r in search_results])
            span.set_attribute("scores", [round(r["score"], 1) for r in search_results])

        # 2. Construction du prompt
        with logfire.span("prompt_building"):
            context_str = formater_contexte(search_results)
            user_prompt = USER_PROMPT_V1.format(context_str=context_str, question=question)

        # 3. Génération + validation (l'agent est tracé automatiquement par instrument_pydantic_ai)
        try:
            resultat = agent.run_sync(user_prompt)
            sortie = resultat.output
            reponse = sortie.reponse
        except Exception:
            logfire.exception("Erreur de l'agent Pydantic AI")
            logging.exception("Erreur de l'agent Pydantic AI")
            sortie, reponse = None, MESSAGE_ERREUR

        # 4. Bilan de la réponse
        logfire.info(
            "Réponse produite",
            information_disponible=sortie.information_disponible if sortie else None,
            nb_sources_citees=len(sortie.sources) if sortie else 0,
            erreur=sortie is None,
        )

    return {
        "answer": reponse,
        "contexts": [r["text"] for r in search_results],
        "sources": search_results,
        "information_disponible": sortie.information_disponible if sortie else None,
        "sources_citees": sortie.sources if sortie else [],
    }


if __name__ == "__main__":
    result = answer("Qui a gagné Roland-Garros en 2025 ?")

    print("\n=== RÉPONSE ===")
    print(result["answer"])
    print(f"\nInformation disponible : {result['information_disponible']}")
    print(f"Sources citées : {result['sources_citees']}")

    print(f"\n=== CONTEXTES RÉCUPÉRÉS ({len(result['contexts'])}) ===")
    for s in result["sources"]:
        print(f"- {s['score']:.1f} % | {s['metadata'].get('source')} | {s['text'][:100]!r}")
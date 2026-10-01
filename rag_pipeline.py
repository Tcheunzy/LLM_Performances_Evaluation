#Fichier contenant toute la logique RAG du prototype afin d'effectuer proprement l'évaluation RAGAS. 
import logging
from utils.config import MISTRAL_API_KEY,MODEL_NAME,SEARCH_K, MISTRAL_RETRY_CONFIG
from utils.vector_store import VectorStoreManager
from mistralai.client import Mistral

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

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
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""


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
def answer(question):
    #On effectue la recherche des chunks
    logging.info(f"Recherche de contexte pour la question : '{question}' avec k={SEARCH_K}")
    search_results = vector_store_manager.search(question, k=SEARCH_K)

    #formatage du contexte et injection dans le prompt
    context_str = formater_contexte(search_results)
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=question)
    messages_for_api = [{"role": "user", "content": final_prompt_for_llm}]
    response_content = generer_reponse(messages_for_api)
    return {"answer":response_content, 
            "contexts": [r["text"] for r in search_results],
            "sources" : search_results,            
    }

if __name__ == "__main__":
    result = answer("Quel est le 3P% de James Harden ?")

    print("\n=== RÉPONSE ===")
    print(result["answer"])

    print(f"\n=== CONTEXTES RÉCUPÉRÉS ({len(result['contexts'])}) ===")
    for s in result["sources"]:
        print(f"- {s['score']:.1f} % | {s['metadata'].get('source')} | {s['text'][:100]!r}")
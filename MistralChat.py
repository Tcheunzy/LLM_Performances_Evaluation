# MistralChat.py (version RAG)
# Migration : mistralai 0.4.2 -> 3.x (client `Mistral`, `chat.complete`, messages en dicts).
# Le prompt et la logique RAG sont ceux du prototype d'origine (baseline de l'audit).
import logging

import streamlit as st
from mistralai.client import Mistral

# --- Importations depuis les modules du projet ---
try:
    from utils.config import (
        APP_TITLE,
        MISTRAL_API_KEY,
        MODEL_NAME,
        NAME,
        SEARCH_K,
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation : {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

# --- Configuration de l'API Mistral ---
model = MODEL_NAME

if not MISTRAL_API_KEY:
    st.error("Erreur : clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()


@st.cache_resource  # Client créé une seule fois, pas à chaque interaction
def get_mistral_client(api_key: str) -> Mistral:
    client = Mistral(api_key=api_key)
    logging.info("Client Mistral initialisé.")
    return client


try:
    client = get_mistral_client(MISTRAL_API_KEY)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()


# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
        st.error("Fichiers d'index ou de chunks non trouvés.")
        st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
        logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager : {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None


vector_store_manager = get_vector_store_manager()

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

# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            f"Bonjour ! Je suis votre analyste IA pour la {NAME}. "
            "Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, "
            "et je vous répondrai en me basant sur les données les plus récentes."
        ),
    }]


# --- Fonctions ---
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
        st.error(f"Erreur lors de l'appel à l'API Mistral : {e}")
        logging.exception("Erreur API Mistral pendant client.chat.complete")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."


def formater_contexte(search_results: list[dict]) -> str:
    """Transforme les résultats de la recherche vectorielle en un bloc de texte pour le prompt."""
    if not search_results:
        return "Aucune information pertinente trouvée dans la base de connaissances pour cette question."

    return "\n\n---\n\n".join(
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\n"
        f"Contenu: {res['text']}"
        for res in search_results
    )


# --- Interface utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle : {model}")

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Vérifier que le Vector Store est disponible
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible pour la recherche.")
        st.stop()

    # 3. Rechercher le contexte dans le Vector Store
    try:
        logging.info(f"Recherche de contexte pour la question : '{prompt}' avec k={SEARCH_K}")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes : {e}")
        logging.exception(f"Erreur pendant vector_store_manager.search pour la query : {prompt}")
        search_results = []  # On continue sans contexte si la recherche échoue

    if not search_results:
        logging.warning(f"Aucun contexte trouvé pour la query : {prompt}")

    # 4. Construire le prompt final
    context_str = formater_contexte(search_results)
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)
    messages_for_api = [{"role": "user", "content": final_prompt_for_llm}]

    # 5. Générer et afficher la réponse
    with st.chat_message("assistant"):
        with st.spinner("L'analyste réfléchit..."):
            response_content = generer_reponse(messages_for_api)
        st.write(response_content)

    # 6. Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- Pied de page ---
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")
# MistralChat.py — interface Streamlit de l'assistant NBA.
# Toute la logique RAG (client Mistral, vector store, prompt, génération) est dans rag_pipeline.py.
# Ce fichier ne gère plus que l'affichage : historique, saisie, appel à answer() et affichage de la réponse.
import logging
 
import streamlit as st
 
from utils.config import APP_TITLE, MODEL_NAME, NAME
 
# --- Chargement du pipeline RAG ---
# L'import crée le client Mistral et charge l'index FAISS, une seule fois :
# Python garde le module en mémoire entre les relances du script par Streamlit,
# donc @st.cache_resource n'est plus nécessaire.
# rag_pipeline lève une RuntimeError si la clé API ou l'index sont absents.
try:
    from rag_pipeline import answer
except Exception as e:
    st.error(f"Impossible de démarrer l'assistant : {e}")
    logging.exception("Erreur au chargement de rag_pipeline")
    st.stop()
 
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
 
# --- Interface utilisateur ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle : {MODEL_NAME}")
 
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
 
    # 2. Interroger le pipeline RAG (recherche -> prompt -> génération) et afficher la réponse
    with st.chat_message("assistant"):
        with st.spinner("L'analyste réfléchit..."):
            try:
                response_content = answer(prompt)["answer"]
            except Exception as e:
                logging.exception(f"Erreur pendant answer() pour la question : {prompt}")
                st.error(f"Une erreur est survenue : {e}")
                response_content = "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."
        st.write(response_content)
 
    # 3. Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response_content})
 
# --- Pied de page ---
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")
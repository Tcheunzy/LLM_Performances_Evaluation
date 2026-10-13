# utils/vector_store.py
# Migration : mistralai 0.4.2 -> 3.x (client `Mistral`, `embeddings.create(inputs=...)`, `MistralError`).
# La logique d'indexation et de recherche est inchangée par rapport au prototype d'origine.
import logging
import os
import pickle
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from .schemas import Chunk, EmbeddedChunk
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # Format attendu par le splitter
from mistralai.client import Mistral
from mistralai.client.errors import MistralError  # Classe mère des erreurs API (SDKError, etc.)

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    MISTRAL_API_KEY,
    MISTRAL_RETRY_CONFIG
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, Any]] = []
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY,retry_config=MISTRAL_RETRY_CONFIG)
        self._load_index_and_chunks()

    # ------------------------------------------------------------------ #
    # Chargement / sauvegarde
    # ------------------------------------------------------------------ #
    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE)):
            logger.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")
            return

        try:
            logger.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
            self.index = faiss.read_index(FAISS_INDEX_FILE)
            logger.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
                self.document_chunks = pickle.load(f)
            logger.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'index/chunks : {e}")
            self.index = None
            self.document_chunks = []

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logger.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logger.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logger.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
                pickle.dump(self.document_chunks, f)
            logger.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'index/chunks : {e}")

    # ------------------------------------------------------------------ #
    # Indexation
    # ------------------------------------------------------------------ #
    def _split_documents_to_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Découpe les documents en chunks validés par Pydantic.

        v1 : les documents « joueur » ne sont PAS redécoupés (1 joueur = 1 chunk, ~1 900 caractères) ;
        les textes (PDF) et tableaux gardent le découpage d'origine (CHUNK_SIZE / CHUNK_OVERLAP).
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks, rejets = [], 0
        for doc_counter, doc in enumerate(documents):
            metadata = doc["metadata"]
            doc_type = metadata.get("doc_type", "texte")

            if doc_type == "joueur":
                morceaux = [(doc["page_content"], 0)]
            else:
                langchain_doc = Document(page_content=doc["page_content"], metadata=metadata)
                morceaux = [
                    (c.page_content, c.metadata.get("start_index", -1))
                    for c in text_splitter.split_documents([langchain_doc])
                ]

            for i, (texte, start_index) in enumerate(morceaux):
                try:
                    chunk = Chunk(id=f"{doc_counter}_{i}", text=texte,
                                  source=metadata["source"], doc_type=doc_type)
                except ValidationError as e:
                    rejets += 1
                    logger.warning(f"Chunk rejeté ({metadata['source']}, n°{i}) : {e.errors()[0]['msg']}")
                    continue
                all_chunks.append({
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": {**metadata, "chunk_id_in_doc": i, "start_index": start_index},
                })

        logger.info(f"{len(all_chunks)} chunks créés, {rejets} rejetés par la validation.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logger.error("Impossible de générer les embeddings : MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logger.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logger.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle : {EMBEDDING_MODEL})...")
        all_embeddings: List = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = i // EMBEDDING_BATCH_SIZE + 1
            texts_to_embed = [chunk["text"] for chunk in chunks[i:i + EMBEDDING_BATCH_SIZE]]
            logger.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")

            try:
                response = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=texts_to_embed,
                )
                all_embeddings.extend(data.embedding for data in response.data)

            except Exception as e:
                # Correction du prototype : toutes les erreurs (API ou autres) sont traitées pareil,
                # pour que le nombre d'embeddings reste égal au nombre de chunks.
                if isinstance(e, MistralError):
                    logger.error(f"Erreur API Mistral (lot {batch_num}) : {e}")
                else:
                    logger.error(f"Erreur inattendue (lot {batch_num}) : {e}")

                if not all_embeddings:
                    logger.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                    continue
                dim = len(all_embeddings[0])
                logger.warning(f"Ajout de {len(texts_to_embed)} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype="float32")] * len(texts_to_embed))

        if not all_embeddings:
            logger.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.array(all_embeddings).astype("float32")
        logger.info(f"Embeddings générés avec succès. Shape : {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, Any]]):
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logger.warning("Aucun document fourni pour construire l'index.")
            return

        # 1. Découper en chunks
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logger.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        # 2. Générer les embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        # 2b. Validation Pydantic : chaque couple (chunk, vecteur) doit être exploitable.
        # On filtre les deux listes ENSEMBLE pour conserver la correspondance « vecteur n°i = chunk n°i ».
        chunks_valides, vecteurs_valides = [], []
        for chunk_dict, vecteur in zip(self.document_chunks, embeddings):
            try:
                EmbeddedChunk(
                    chunk=Chunk(id=chunk_dict["id"], text=chunk_dict["text"],
                                source=chunk_dict["metadata"]["source"],
                                doc_type=chunk_dict["metadata"].get("doc_type", "texte")),
                    embedding=vecteur.tolist(),
                )
            except ValidationError as e:
                logger.warning(f"Embedding rejeté (chunk {chunk_dict['id']}) : {e.errors()[0]['msg']}")
                continue
            chunks_valides.append(chunk_dict)
            vecteurs_valides.append(vecteur)

        if not chunks_valides:
            logger.error("Aucun embedding valide : index non créé.")
            return
        self.document_chunks = chunks_valides
        embeddings = np.array(vecteurs_valides, dtype="float32")
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logger.error("Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            self.document_chunks = []
            self.index = None
            for path in (FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE):
                if os.path.exists(path):
                    os.remove(path)
            return

        # 3. Index Faiss pour la similarité cosinus (vecteurs normalisés + produit scalaire)
        dimension = embeddings.shape[1]
        logger.info(f"Création de l'index Faiss (cosinus) avec dimension {dimension}...")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logger.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. Sauvegarder
        self._save_index_and_chunks()

    # ------------------------------------------------------------------ #
    # Recherche
    # ------------------------------------------------------------------ #
    def search(self, query_text: str, k: int = 5, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: texte de la requête
            k: nombre de résultats à retourner
            min_score: score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste de dicts {score, raw_score, text, metadata}, triée par score décroissant.
        """
        if self.index is None or not self.document_chunks:
            logger.warning("Recherche impossible : l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logger.error("Recherche impossible : MISTRAL_API_KEY manquante pour l'embedding de la requête.")
            return []

        logger.info(f"Recherche des {k} chunks les plus pertinents pour : '{query_text}'")
        try:
            # 1. Embedding de la requête
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text],
            )
            query_embedding = np.array([response.data[0].embedding]).astype("float32")
            faiss.normalize_L2(query_embedding)

            # 2. Recherche (plus de candidats si un filtre de score est appliqué)
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formatage des résultats
            min_score_percent = min_score * 100 if min_score is not None else None
            results = []
            for raw_score, idx in zip(scores[0], indices[0]):
                if not 0 <= idx < len(self.document_chunks):
                    logger.warning(f"Index Faiss {idx} hors limites (taille des chunks : {len(self.document_chunks)}).")
                    continue

                similarity = float(raw_score) * 100  # Cosinus -> pourcentage
                if min_score_percent is not None and similarity < min_score_percent:
                    logger.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                    continue

                chunk = self.document_chunks[idx]
                results.append({
                    "score": similarity,
                    "raw_score": float(raw_score),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:k]
            logger.info(f"{len(results)} chunks pertinents trouvés.")
            return results

        except MistralError as e:
            logger.error(f"Erreur API Mistral lors de l'embedding de la requête : {e}")
            return []
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la recherche : {e}")
            return []
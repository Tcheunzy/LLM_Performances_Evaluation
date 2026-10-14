# diag_retrieval.py — diagnostic : a-t-on cassé quelque chose, ou le prototype échoue-t-il par conception ?
# Lancement : uv run python diag_retrieval.py
import numpy as np

from utils.config import EMBEDDING_MODEL
from utils.vector_store import VectorStoreManager
import sys
from datetime import datetime
from pathlib import Path

class Tee:
    """Duplique ce qui est écrit : terminal + fichier."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

VERSION = "v0"  # à changer à chaque version évaluée (v1, v2...)
OUTPUT = Path("docs/audit") / f"diag_retrieval_{VERSION}_{datetime.now():%Y%m%d_%H%M}.txt"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

log_file = open(OUTPUT, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

QUESTION = "Quel est le 3P% de James Harden ?"
m = VectorStoreManager()

# --- Test 1 : intégrité -------------------------------------------------------
# On recalcule aujourd'hui l'embedding de quelques chunks et on le compare au vecteur
# stocké dans l'index d'origine. Cosinus ≈ 1.00 → même modèle, index intact, migration neutre.
print("\n=== Test 1 : intégrité de l'index (cosinus attendu ≈ 1.00)")
for i in (0, 150, 300):
    stored = m.index.reconstruct(i)
    resp = m.mistral_client.embeddings.create(model=EMBEDDING_MODEL, inputs=[m.document_chunks[i]["text"]])
    fresh = np.array(resp.data[0].embedding, dtype="float32")
    fresh /= np.linalg.norm(fresh)
    print(f"  chunk {i:>3} ({m.document_chunks[i]['metadata']['source']}) : cosinus = {float(fresh @ stored):.4f}")

# --- Test 2 : ce que le retriever renvoie réellement ----------------------------
print(f"\n=== Test 2 : top-5 pour « {QUESTION} »")
for r in m.search(QUESTION, k=5):
    print(f"  {r['score']:5.1f} %  {r['metadata']['source']:<45}  contient 'Harden' : {'Harden' in r['text']}")

# --- Test 3 : où se classe le chunk qui contient la bonne réponse ? ---------------
target = next(
    i for i, c in enumerate(m.document_chunks)
    if "James Harden" in c["text"]
    and ("Données NBA" in c["metadata"]["source"] or "Joueur: James Harden" in c["metadata"]["source"])
)
ranking = m.search(QUESTION, k=len(m.document_chunks))
rank = next(pos for pos, r in enumerate(ranking, 1) if r["text"] == m.document_chunks[target]["text"])
score = ranking[rank - 1]["score"]
print(f"\n=== Test 3 : le chunk avec la ligne de Harden (n°{target}) est classé {rank}e / {len(ranking)} (score {score:.1f} %)")
print("  (le LLM ne voit que les 5 premiers)")
line = next(l for l in m.document_chunks[target]["text"].splitlines() if "Harden" in l)
print(f"  Ce que contient ce chunk pour Harden :\n  {line.strip()[:160]}...")


sys.stdout = sys.__stdout__  # on rétablit la sortie normale
log_file.close()
print(f"\nRésultats enregistrés dans {OUTPUT}")
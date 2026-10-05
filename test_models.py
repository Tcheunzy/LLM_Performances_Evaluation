# test_modeles.py — quels modèles Mistral mon abonnement autorise-t-il ? (choix du juge RAGAS)
# Lancement : uv run test_modeles.py
import time

from mistralai.client import Mistral

from utils.config import MISTRAL_API_KEY

client = Mistral(api_key=MISTRAL_API_KEY)  # volontairement SANS relance : on veut le statut brut

# 1. Modèles visibles par la clé (la liste ne garantit pas l'accès : on teste ensuite)
ids = sorted(m.id for m in client.models.list().data)
print("Modèles listés :", ", ".join(i for i in ids if any(k in i for k in ("large", "medium", "small", "ministral", "magistral"))))

# 2. Un appel minimal par candidat, du plus puissant au plus léger
CANDIDATS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "magistral-medium-latest",
    "mistral-small-latest",
    "mistral-small-2506",
    "ministral-14b-latest",
    "ministral-8b-latest",
]
print("\nTest d'accès :")
for modele in CANDIDATS:
    try:
        client.chat.complete(model=modele, messages=[{"role": "user", "content": "OK"}], max_tokens=3)
        statut = "✅ accessible"
    except Exception as e:
        code = getattr(e, "status_code", "?")
        statut = {403: "⛔ refusé par l'abonnement (403)", 404: "❌ inexistant (404)",
                  429: "⚠️ accessible mais limite de débit (429)"}.get(code, f"erreur {code}")
    print(f"  {modele:<26} {statut}")
    time.sleep(2)  # on espace les appels pour ne pas déclencher nous-mêmes des 429
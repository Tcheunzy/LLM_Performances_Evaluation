# SportSee — Assistant NBA (RAG + SQL) et son évaluation

Assistant conversationnel pour les coachs et analystes de SportSee. Il répond en français aux questions sur la
saison régulière NBA 2024-25 en combinant deux sources :

- **des documents** (discussions Reddit en PDF), retrouvés par recherche sémantique (FAISS) ;
- **une base SQL** des statistiques des 569 joueurs, interrogée par un outil de génération de requêtes.

Le projet contient aussi tout le **protocole d'évaluation** (RAGAS + exactitude contrôlée, 50 questions en
6 catégories) qui a guidé les améliorations du prototype initial jusqu'à la version **v2b**.
Le rapport complet est dans [`docs/rapport/rapport_evaluation.md`](docs/rapport/rapport_evaluation.md).

## Résultats en bref

| Version | Changement principal | Simples exactes | Bruitées | Multicritères | Refus corrects | Erreurs |
|---|---|---|---|---|---|---|
| v0 | prototype fourni | 0/12 | 2/8 | 4/12 | 0/10 | 0 |
| v1b | données corrigées, prompt, agent Pydantic AI | 10/12 | 4/8 | 1/12 | 10/10 | 0 |
| **v2b** | **outil SQL + correctifs** | **11/12** | **5/8** | **10/12** | **10/10** | **0** |

Le détail (toutes les versions, scores RAGAS, analyse des échecs et limites) est dans le rapport.

---

## Sommaire

1. [Architecture](#architecture)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Mise en place du système](#mise-en-place-du-système)
5. [Utiliser l'API](#utiliser-lapi)
6. [Évaluer l'assistant](#évaluer-lassistant)
7. [Arborescence](#arborescence)
8. [Journalisation et observabilité](#journalisation-et-observabilité)
9. [Sécurité](#sécurité)
10. [Accessibilité](#accessibilité)
11. [Limites connues](#limites-connues)

---

## Architecture

![Schéma d'architecture de la v2b. Hors ligne : les PDF sont découpés et indexés dans FAISS ; l'Excel est validé par Pydantic puis chargé dans SQLite. En ligne : la question est envoyée à la recherche FAISS, puis à un agent Pydantic AI qui peut appeler l'outil SQL ; la réponse structurée est validée avant d'être renvoyée par l'API.](docs/rapport/images/architecture.png)

**Hors ligne (une fois, via `setup_rag.py`)**

1. Les 4 PDF de `inputs/` sont lus (OCR si nécessaire), découpés en extraits et vectorisés avec `mistral-embed`,
   puis stockés dans un index **FAISS** (similarité cosinus, 644 extraits).
2. Le fichier `inputs/regular NBA.xlsx` est lu ligne par ligne et **validé par Pydantic** (`utils/schemas.py`) :
   types, bornes, cohérence entre colonnes. Les données valides sont chargées dans une base **SQLite**
   (`data/nba.db` : tables `teams`, `players`, `player_stats`, `matches` vide, vue `player_per_game`).
   Les anomalies sont consignées dans `docs/audit/anomalies_donnees.csv`.

**En ligne (à chaque question)**

1. Recherche des 5 extraits les plus proches dans FAISS.
2. Un **agent Pydantic AI** (`ministral-3b-2512`) reçoit la question et ces extraits. Pour un calcul, un
   classement ou une comparaison, il appelle l'outil **`interroger_statistiques`** (`sql_tool.py`) :
   - génération d'une requête SQL par LangChain avec 9 exemples *few-shot* et le schéma décrit avec ses unités ;
   - vérification (une seule instruction `SELECT`/`WITH`, mots-clés interdits) puis exécution sur une
     connexion **en lecture seule**.
3. La réponse est structurée (`réponse`, `information_disponible`, `sources`) et **validée** : une source citée
   doit faire partie des documents récupérés ou de la base SQL.
4. L'API FastAPI renvoie la réponse, les sources citées et la trace des requêtes SQL.

---

## Prérequis

- **Python 3.14** et **[uv](https://docs.astral.sh/uv/)** (recommandé) ou `pip`.
- Une **clé API Mistral** ([console.mistral.ai](https://console.mistral.ai)).
- Environ **1,3 Go** d'espace disque pour l'installation complète, dont PyTorch (installé par `easyocr`
  pour l'OCR). L'API seule n'en a pas besoin (voir ci-dessous).

## Installation

```bash
git clone <url-du-dépôt>
cd LLM_Performances_Evaluation
uv sync
```

`uv sync` installe les dépendances principales et les groupes `dev`, `indexing` et `eval`
(définis dans `pyproject.toml`). Autres cas :

| Usage | Commande |
|---|---|
| Assistant + API uniquement (sans PyTorch ni RAGAS) | `uv sync --no-default-groups` |
| Avec `pip` (tout sauf les outils de développement) | `pip install -r requirements.txt` |

Créer ensuite le fichier `.env` à partir du modèle :

```bash
cp .env.example .env        # Windows PowerShell : Copy-Item .env.example .env
```

| Variable | Rôle | Exemple |
|---|---|---|
| `MISTRAL_API_KEY` | clé de l'API Mistral (obligatoire) | `xxxxxxxx` |
| `VECTOR_DB_DIR` | dossier de l'index FAISS utilisé | `vector_db_v1` |

Logfire (traces) est facultatif : sans jeton Logfire configuré, rien n'est envoyé.

---

## Mise en place du système

Un seul lancement met le système en état de marche :

```bash
uv run setup_rag.py
```

Le script enchaîne quatre étapes, chacune journalisée avec sa durée, et s'arrête à la première en échec
(code de sortie 1) :

| Étape | Rôle |
|---|---|
| `environnement` | vérifie la clé API et les fichiers d'entrée, avant tout traitement long |
| `index` | construit l'index FAISS ; **réutilise l'index existant** s'il est présent |
| `base` | reconstruit `data/nba.db` à partir de l'Excel validé et contrôle les comptes |
| `test` | pose deux questions de contrôle : une à laquelle l'assistant doit répondre, une qu'il doit refuser |

Options :

```bash
uv run setup_rag.py --etapes base test     # seulement certaines étapes
uv run setup_rag.py --force-index          # reconstruire l'index (OCR + embeddings : plusieurs minutes)
```

> ⚠️ `--force-index` écrase l'index pointé par `VECTOR_DB_DIR`. L'index `vector_db_v1/` versionné est celui
> de toutes les évaluations : pour expérimenter, pointer `VECTOR_DB_DIR` vers un nouveau dossier.

Exemple de bilan :

```text
2026-10-13 14:36:25,739 | INFO    | setup_rag | --- Bilan ---
2026-10-13 14:36:25,739 | INFO    | setup_rag | environnement  ok        1.9 s
2026-10-13 14:36:25,739 | INFO    | setup_rag | index          ok        0.0 s
2026-10-13 14:36:25,740 | INFO    | setup_rag | base           ok        3.2 s
2026-10-13 14:36:25,740 | INFO    | setup_rag | test           ok        3.9 s
2026-10-13 14:36:25,740 | INFO    | setup_rag | Système opérationnel. Journal complet : logs/setup_rag.log
```

---

## Utiliser l'API

```bash
uv run uvicorn api:app --reload
```

La documentation interactive (Swagger UI) est alors disponible sur **http://127.0.0.1:8000/docs** :
chaque route y est décrite et peut être essayée depuis le navigateur.

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/ask` | pose une question à l'assistant |
| `GET` | `/health` | état du service (index chargé, base accessible) |
| `GET` | `/docs` | documentation interactive |

### `POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les 3 meilleurs marqueurs par match ?"}'
```

Sous Windows PowerShell :

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json; charset=utf-8" `
  -Body '{"question": "Quels sont les 3 meilleurs marqueurs par match ?"}' | ConvertTo-Json -Depth 5
```

Réponse (200) :

```json
{
  "question": "Quels sont les 3 meilleurs marqueurs par match ?",
  "reponse": "Les trois meilleurs marqueurs par match selon les données disponibles sont :\n1. Shai Gilgeous-Alexander (Oklahoma City Thunder) avec **32,7 points par match**,\n2. Giannis Antetokounmpo (Milwaukee Bucks) avec **30,4 points par match**,\n3. Nikola Jokić (Denver Nuggets) avec **29,6 points par match**.",
  "information_disponible": true,
  "sources": ["Base de données NBA (SQL)"],
  "requetes_sql": [
    {
      "question": "Quels sont les trois joueurs avec la moyenne de points marqués par match les plus élevées lors de la saison régulière NBA 2024-25 ?",
      "requete": "SELECT name, team_name, pts_per_game FROM player_per_game ORDER BY pts_per_game DESC LIMIT 3",
      "nb_lignes": 3,
      "erreur": null
    }
  ],
  "duree_s": 2.51
}
```

| Champ | Signification |
|---|---|
| `reponse` | texte de la réponse, en français |
| `information_disponible` | `false` si l'assistant indique que l'information n'est pas dans ses données (question hors NBA, donnée absente) |
| `sources` | sources citées : nom d'un document, ou `Base de données NBA (SQL)` |
| `requetes_sql` | requêtes exécutées par l'outil SQL (vide si l'assistant ne l'a pas utilisé) |
| `duree_s` | temps de traitement côté serveur |

Codes de retour :

| Code | Cas |
|---|---|
| `200` | réponse produite, **y compris un refus justifié** (`information_disponible: false`) |
| `422` | requête invalide (question absente, vide, ou de plus de 500 caractères) |
| `503` | échec technique (LLM indisponible) ; `/health` : base ou index inaccessible |

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"statut": "ok", "index_vectoriel": 644, "joueurs_en_base": 569, "version": "2b"}
```

---

## Évaluer l'assistant

Le jeu de test (`eval/testset.json`) contient **50 questions** avec leur réponse de référence, en 6 catégories :
simples, bruitées (fautes, formulations familières), multicritères, textuelles, hors données, hors sujet.

```bash
uv run evaluate_ragas.py --version v2b              # les 50 questions (~30 min, juge ministral-14b-2512)
uv run evaluate_ragas.py --version essai --limit 5  # essai rapide
```

Les résultats sont écrits dans `eval/results/<version>/` :

| Fichier | Contenu |
|---|---|
| `reponses.json` | question, réponse, contextes, sources, requêtes SQL, durée, erreur éventuelle |
| `scores_par_question.csv` / `scores_par_categorie.csv` | métriques RAGAS (faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness) |
| `bilan_seuils.csv` | comparaison aux seuils minimum (bêta) et cible (production) |
| `config.json` | modèles, index et description du pipeline évalués (traçabilité) |
| `evaluation.log` | journal de l'exécution |

Outils d'analyse (dossier `scripts/`), à lancer **depuis la racine** sous forme de module :

```bash
uv run python -m scripts.comparer_versions --versions v2 v2b
```

| Script | Rôle |
|---|---|
| `scripts.comparer_versions --versions A B` | compare deux versions **sur les mêmes questions** (`--metrique`, `--categorie`, `--ecarts`) |
| `scripts.analyse_refus --version V` | liste les réponses aux questions hors données / hors sujet et le taux de refus déclaré |
| `scripts.voir_reponses --version V --ids S01 M04` | affiche les réponses de certaines questions |
| `scripts.origine_reponses` | vérifie si chaque élément attendu figure dans la réponse **et** dans le contexte récupéré (donnée issue du contexte ou du modèle ?) |
| `scripts.debug_erreurs` | rejoue une question et affiche les échanges avec le modèle (appels d'outil, relances de validation) |
| `scripts.diagnostic_retrieval_prototype` | diagnostic de la recherche documentaire du prototype (v0) |
| `scripts.verifier_acces_modeles` | liste les modèles Mistral disponibles et teste l'accès de la clé à chacun (403 = hors abonnement, 429 = limite de débit) |
| `scripts.build_testset`, `scripts.build_elements_attendus` | construction du jeu de test et des éléments attendus pour l'exactitude |
| `utils.bilan_seuils --version V` | recalcule le bilan par rapport aux seuils d'une version déjà évaluée |

> Les scores RAGAS varient d'une exécution à l'autre (écarts mesurés jusqu'à 0,5 sur une question entre deux
> exécutions identiques de la v0). Les comparaisons s'appuient donc aussi sur l'**exactitude**, vérifiée de
> façon déterministe à partir des éléments attendus (`eval/elements_attendus.json`, `utils/exactitude.py`).

---

## Arborescence

```text
.
├── setup_rag.py               # mise en place complète en un lancement
├── api.py                     # API FastAPI (POST /ask, GET /health)
├── rag_pipeline.py            # pipeline : recherche FAISS + agent Pydantic AI + validation
├── sql_tool.py                # outil SQL : génération few-shot (LangChain) et exécution en lecture seule
├── load_excel_to_db.py        # Excel -> validation Pydantic -> SQLite + journal des anomalies
├── indexer.py                 # PDF -> extraits -> embeddings -> index FAISS
├── evaluate_ragas.py          # évaluation RAGAS d'une version
├── utils/
│   ├── config.py              # modèles, chemins, paramètres
│   ├── schemas.py             # modèles Pydantic (joueurs, équipes, réponse de l'assistant)
│   ├── db_schema.py           # schéma SQL (SQLAlchemy), descriptions des colonnes
│   ├── agent_outils.py        # outil interroger_statistiques et contexte de l'agent
│   ├── vector_store.py        # index FAISS : découpage, embeddings, recherche
│   ├── data_loader.py         # lecture des PDF (dont OCR)
│   ├── bilan_seuils.py        # comparaison des scores aux seuils (importé par evaluate_ragas)
│   └── nettoyage_*.py, exactitude.py, formatage_joueur.py
├── scripts/                   # outils d'analyse et de construction du jeu de test (voir « Évaluer »)
├── inputs/                    # données sources : 4 PDF Reddit + regular NBA.xlsx
├── vector_db/                 # index du prototype (v0)
├── vector_db_v1/              # index des versions v1 et v2
├── eval/
│   ├── testset.json           # 50 questions de test et leurs références
│   ├── elements_attendus.json # éléments vérifiés pour l'exactitude
│   └── results/<version>/     # résultats de chaque version évaluée
├── docs/
│   ├── rapport/               # rapport d'évaluation (.md) et ses figures (images/)
│   └── audit/                 # journal des anomalies de données, diagnostics de recherche
├── pyproject.toml, uv.lock    # dépendances (uv)
├── requirements.txt           # dépendances figées (pip)
└── .env.example               # modèle du fichier de configuration
```

Fichiers générés, non versionnés : `.env`, `data/nba.db`, `logs/`.

---

## Journalisation et observabilité

- **Journal structuré** : chaque point d'entrée (`setup_rag.py`, `api.py`, `evaluate_ragas.py`,
  `load_excel_to_db.py`, `indexer.py`) configure le même format, et chaque module écrit sous son propre nom :
  ```text
  2026-10-13 15:15:20,687 | INFO    | utils.vector_store | Chargement de l'index Faiss depuis vector_db_v1\faiss_index.idx...
  ```
  Fichiers : `logs/setup_rag.log`, `eval/results/<version>/evaluation.log`.
- **Durées** : par étape de mise en place, par question (API et évaluation), et pour la notation RAGAS.
- **Logfire** : chaque question produit une trace (`answer` → `retrieval` → `prompt_building` → `agent run`,
  avec les appels au modèle et à l'outil SQL), si un jeton Logfire est configuré.

## Sécurité

- La clé API est lue dans `.env`, **jamais versionné** (`.gitignore`) ; `.env.example` ne contient aucun secret.
- Les requêtes générées par le LLM sont exécutées sur une connexion SQLite **en lecture seule**, après un
  contrôle (une seule instruction, `SELECT`/`WITH`, mots-clés d'écriture refusés) et avec un nombre de lignes
  borné : une requête malveillante ou erronée ne peut pas modifier la base.
- Les entrées de l'API sont validées (longueur de la question) avant tout appel au modèle.

## Accessibilité

- Ce README et le rapport sont structurés par titres hiérarchiques, avec un sommaire, et chaque image a un
  **texte alternatif** décrivant son contenu ; les tableaux ont des en-têtes.
- Aucune information n'est portée par la couleur seule : les statuts sont écrits en toutes lettres
  (`ok`, `échec`, `cible`, `non atteint`).
- L'assistant répond en texte simple, en français, et signale explicitement quand l'information n'est pas
  disponible plutôt que de rester silencieux ou d'inventer.
- L'API renvoie du JSON en UTF-8 déclaré, exploitable par tout client, y compris un lecteur d'écran via
  `/docs` ou une interface tierce.

## Limites connues

- **Une saison, des totaux** : l'Excel ne contient que les totaux de la saison régulière 2024-25 ; aucune
  question par match ou par période (« ses 5 derniers matchs ») n'est possible. La table `matches` existe
  mais est vide, en attente de données.
- **Petit modèle** : `ministral-3b` choisit parfois mal entre documents et SQL, ou ignore une règle du prompt ;
  toute modification du prompt impose de réévaluer l'ensemble du jeu de test.
- **Questions textuelles** (contenu des discussions Reddit) : c'est la catégorie la plus faible ; elles ne
  relèvent pas de l'outil SQL.
- **Évaluation** : 50 questions, une exécution par version (deux pour la v0), juge LLM bruité. Voir la
  section « Analyse critique » du rapport.

# debug_erreurs.py — pourquoi l'agent échoue-t-il à la validation sur certaines questions ?
#
# Rejoue les questions et affiche, pour chacune, les réponses proposées par le modèle (APPEL)
# et les messages d'erreur du validateur qui lui ont été renvoyés (RELANCE).
# Lancement : uv run debug_erreurs.py T03 T06 T08
import json
import sys
from pathlib import Path

from pydantic_ai import capture_run_messages
from pydantic_ai.messages import RetryPromptPart, ToolCallPart

from rag_pipeline import answer

ids = set(sys.argv[1:])
testset = json.loads((Path(__file__).resolve().parents[1] / "eval" / "testset.json").read_text(encoding="utf-8"))

for q in [q for q in testset if q["id"] in ids]:
    print(f"\n===== [{q['id']}] {q['question']}")
    with capture_run_messages() as messages:  # enregistre tous les échanges de l'agent pendant answer()
        resultat = answer(q["question"])
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"  APPEL   {part.tool_name} : {str(part.args)[:400]}")
            elif isinstance(part, RetryPromptPart):
                erreurs = part.content if isinstance(part.content, list) else [part.content]
                for e in erreurs:
                    print(f"  RELANCE {e['msg'] if isinstance(e, dict) else e}")
    print(f"  RÉSULTAT : {resultat['answer'][:150]}")
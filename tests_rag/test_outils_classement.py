# tests/test_outils_classement.py — tests des outils déterministes de classement (sans appel à l'API).
# Les valeurs attendues sont celles de l'Excel (vérifiées avec pandas). Nécessite data/nba.db.
# Lancement (depuis la racine) : uv run python -m tests_rag.test_outils_classement
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from utils.agent_outils import MAX_APPELS_SQL, SOURCE_SQL, ContexteRecherche
from utils.outils_classement import comparer_a_un_seuil, rang_joueur


def contexte():
    """Imite le RunContext de Pydantic AI : seul ctx.deps est utilisé par les outils."""
    return SimpleNamespace(deps=ContexteRecherche(sources_autorisees=[], contextes=[]))


def test_rang_calcule_sur_toute_la_ligue():
    # Le défaut corrigé : avant, le filtre sur le joueur précédait le classement -> toujours « 1er »
    assert "Rang : 2 sur" in rang_joueur(contexte(), "fta", "Shai Gilgeous-Alexander")
    assert "Rang : 3 sur" in rang_joueur(contexte(), "tov", "Cade Cunningham")
    assert "Rang : 1 sur" in rang_joueur(contexte(), "three_pm", "Anthony Edwards")


def test_comparaison_a_un_seuil():
    assert "OUI" in comparer_a_un_seuil(contexte(), "three_pct", "Cade Cunningham", 35)   # 35,6 > 35
    assert "NON" in comparer_a_un_seuil(contexte(), "three_pct", "Julius Randle", 35)     # 34,4 < 35


def test_joueur_ambigu_ou_absent():
    assert "Plusieurs joueurs" in rang_joueur(contexte(), "pts", "Wagner")       # Franz et Moritz
    assert "Aucun joueur" in rang_joueur(contexte(), "pts", "Reggie Miller")    # retraité


def test_tracabilite():
    ctx = contexte()
    rang_joueur(ctx, "fta", "Shai Gilgeous-Alexander")
    assert SOURCE_SQL in ctx.deps.sources_autorisees     # le validateur acceptera la source
    assert len(ctx.deps.contextes) == 1                   # RAGAS verra le résultat
    assert len(ctx.deps.requetes_sql) == 1                # trace dans reponses.json
    ctx_absent = contexte()
    rang_joueur(ctx_absent, "pts", "Reggie Miller")
    assert ctx_absent.deps.sources_autorisees == []      # un échec n'est pas une source citable



def test_nom_abime_des_fiches():
    # Les fiches FAISS contiennent des noms non nettoyés (« P,J, Washington ») : l'outil les nettoie
    assert "P.J. Washington" in rang_joueur(contexte(), "pts", "P,J, Washington")


def test_limite_stricte_meme_en_parallele():
    # En v2d, 40 appels parallèles avaient contourné la limite : chacun lisait le compteur avant les autres
    ctx = contexte()
    with ThreadPoolExecutor(max_workers=10) as pool:
        reponses = list(pool.map(lambda _: rang_joueur(ctx, "pts", "Trae Young"), range(10)))
    acceptes = [r for r in reponses if not r.startswith("Limite")]
    assert len(acceptes) == MAX_APPELS_SQL, len(acceptes)


if __name__ == "__main__":
    tests = [f for nom, f in dict(globals()).items() if nom.startswith("test_")]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"{len(tests)} tests réussis")
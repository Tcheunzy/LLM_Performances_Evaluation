# utils/nettoyage_donnees.py — nettoyage des données tabulaires (feuille « Données NBA ») avant la base SQL.
#
# Anomalie corrigée : dans l'Excel source, les points des noms ont été remplacés par des virgules
# (31 joueurs) : « Jaren Jackson Jr, », « P,J, Washington ». Aucun nom NBA ne contient de virgule
# légitime, donc la correction est certaine : chaque virgule redevient un point.
# Les autres anomalies (REB ≠ OREB + DREB) ne sont PAS corrigées ici : on ne sait pas quelle valeur
# est fausse. Elles seront signalées par le modèle Pydantic (utils/schemas.py).
import unicodedata


def nettoyer_nom_joueur(nom: str) -> str:
    """Corrige un nom de joueur : virgules -> points, espaces superflus, Unicode normalisé.

    >>> nettoyer_nom_joueur("P,J, Washington")
    'P.J. Washington'
    >>> nettoyer_nom_joueur("Jaren Jackson Jr,")
    'Jaren Jackson Jr.'
    >>> nettoyer_nom_joueur("  Nikola   Jokić ")
    'Nikola Jokić'
    """
    if not isinstance(nom, str):
        raise ValueError(f"Nom de joueur invalide : {nom!r}")
    nom = unicodedata.normalize("NFC", nom)  # « Jokić » toujours encodé de la même façon
    nom = nom.replace(",", ".")
    nom = " ".join(nom.split())              # espaces en trop, en début ou en fin
    if not nom:
        raise ValueError("Nom de joueur vide.")
    return nom


if __name__ == "__main__":
    # Test : uv run python -m utils.nettoyage_donnees
    import doctest
    from pathlib import Path

    import pandas as pd

    # 1. Les exemples de la docstring
    resultat = doctest.testmod()
    print(f"Doctests : {resultat.attempted - resultat.failed}/{resultat.attempted} réussis")

    # 2. Sur les vrais noms de l'Excel
    excel = Path(__file__).resolve().parent.parent / "inputs" / "regular NBA.xlsx"
    noms = pd.read_excel(excel, sheet_name="Données NBA", header=1)["Player"]
    corriges = noms.map(nettoyer_nom_joueur)
    modifies = noms[noms != corriges]

    print(f"{len(modifies)} noms corrigés sur {len(noms)} :")
    for avant, apres in zip(modifies, corriges[modifies.index]):
        print(f"  {avant!r:<28} -> {apres!r}")
    print(f"Doublons après nettoyage : {corriges.duplicated().sum()}")
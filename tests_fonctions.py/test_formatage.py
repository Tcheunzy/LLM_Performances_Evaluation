# test_formatage.py — vérifie la validation et le formatage d'un joueur
import pandas as pd

from utils.formatage_joueur import joueur_vers_texte
from utils.schemas import PlayerStats

EXCEL = "inputs/regular NBA.xlsx"

# Même préparation que dans build_testset.py
df = pd.read_excel(EXCEL, sheet_name="Données NBA", header=1)
df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
df = df.rename(columns={[c for c in df.columns if not isinstance(c, str)][0]: "3PM"})

equipes = pd.read_excel(EXCEL, sheet_name="Equipe")
noms_equipes = dict(zip(equipes.iloc[:, 0], equipes.iloc[:, 1]))

# Un joueur précis
ligne = df[df["Player"] == "James Harden"].iloc[0].to_dict()

joueur = PlayerStats.model_validate(ligne)    # 1. contrôle Pydantic
texte = joueur_vers_texte(ligne, noms_equipes)  # 2. formatage

print(f"Validé : {joueur.player} ({joueur.team}), 3P% = {joueur.three_pct}\n")
print(texte)
print(f"\nLongueur : {len(texte)} caractères")
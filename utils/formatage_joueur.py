# utils/formatage_joueur.py — transforme une ligne de joueur (feuille « Données NBA ») en texte lisible.
#
# Choix de conception (v1) : une ligne de joueur = un document, avec les 45 colonnes (aucune perte
# d'information) et des libellés explicites en français, pour rapprocher le texte du vocabulaire
# des questions et lever les ambiguïtés du dictionnaire des données d'origine.
#
# Libellés vérifiés sur les données (et non repris tels quels du dictionnaire, qui est erroné) :
#   - PTS, REB, AST, FGM, 3PA, POSS... sont des TOTAUX sur la saison (ex. Harden : 1801 points en 79 matchs) ;
#   - Min et +/- sont des MOYENNES par match ;
#   - « 3PM » est le nombre de tirs à 3 points réussis (le dictionnaire, trompé par la conversion
#     de l'en-tête en 15:00:00 par Excel, le décrit à tort comme des « minutes jouées après 15:00 »).

LIBELLES = {
    "Player": "Joueur",
    "Team": "Équipe",
    "Age": "Âge",
    "GP": "Matchs joués (GP)",
    "W": "Victoires de l'équipe lors des matchs joués (W)",
    "L": "Défaites de l'équipe lors des matchs joués (L)",
    "Min": "Minutes jouées par match, en moyenne (Min)",
    "PTS": "Points marqués, total saison (PTS)",
    "FGM": "Tirs réussis, total saison (FGM)",
    "FGA": "Tirs tentés, total saison (FGA)",
    "FG%": "Pourcentage de réussite aux tirs (FG%)",
    "3PM": "Tirs à 3 points réussis, total saison (3PM)",
    "3PA": "Tirs à 3 points tentés, total saison (3PA)",
    "3P%": "Pourcentage de réussite à 3 points (3P%)",
    "FTM": "Lancers francs réussis, total saison (FTM)",
    "FTA": "Lancers francs tentés, total saison (FTA)",
    "FT%": "Pourcentage de réussite aux lancers francs (FT%)",
    "OREB": "Rebonds offensifs, total saison (OREB)",
    "DREB": "Rebonds défensifs, total saison (DREB)",
    "REB": "Rebonds, total saison (REB)",
    "AST": "Passes décisives, total saison (AST)",
    "TOV": "Balles perdues, total saison (TOV)",
    "STL": "Interceptions, total saison (STL)",
    "BLK": "Contres, total saison (BLK)",
    "PF": "Fautes personnelles, total saison (PF)",
    "FP": "Points fantasy, total saison (FP)",
    "DD2": "Double-doubles (DD2)",
    "TD3": "Triple-doubles (TD3)",
    "+/-": "Plus-minus moyen par match (+/-)",
    "OFFRTG": "Rating offensif (OFFRTG)",
    "DEFRTG": "Rating défensif (DEFRTG)",
    "NETRTG": "Net rating (NETRTG)",
    "AST%": "Pourcentage de passes décisives (AST%)",
    "AST/TO": "Ratio passes décisives / balles perdues (AST/TO)",
    "AST RATIO": "Ratio de passes décisives pour 100 possessions (AST RATIO)",
    "OREB%": "Pourcentage de rebonds offensifs (OREB%)",
    "DREB%": "Pourcentage de rebonds défensifs (DREB%)",
    "REB%": "Pourcentage de rebonds (REB%)",
    "TO RATIO": "Balles perdues pour 100 possessions (TO RATIO)",
    "EFG%": "Pourcentage de tirs effectif (EFG%)",
    "TS%": "True shooting (TS%)",
    "USG%": "Taux d'utilisation (USG%)",
    "PACE": "Rythme de jeu (PACE)",
    "PIE": "Player Impact Estimate (PIE)",
    "POSS": "Possessions jouées, total saison (POSS)",
}

def formater_valeur(valeur) -> str:
    """Nombre au format français ; 41.0 devient 41, 35.2 devient 35,2."""
    if isinstance(valeur, float) and valeur.is_integer():
        valeur = int(valeur)
    return str(valeur).replace(".", ",")

def joueur_vers_texte(ligne: dict, noms_equipes: dict) -> str:
    """Construit le texte d'un joueur à partir de la ligne Excel COMPLÈTE (les 45 colonnes).

    ligne        : dictionnaire {nom de colonne Excel: valeur}, déjà validé par PlayerStats.
    noms_equipes : correspondance code -> nom complet (feuille « Equipe »).
    """
    morceaux = []
    for colonne, libelle in LIBELLES.items():
        valeur = ligne[colonne]
        if colonne == "Team":
            texte_valeur = f"{valeur} ({noms_equipes.get(valeur, 'nom inconnu')})"
        else:
            texte_valeur = formater_valeur(valeur)
        morceaux.append(f"{libelle} : {texte_valeur}")
    return "Statistiques de saison régulière NBA — " + " | ".join(morceaux)
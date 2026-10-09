# utils/schemas.py — modèles Pydantic : le « contrat » de chaque donnée du pipeline
from pydantic import BaseModel, ConfigDict, Field, model_validator
import unicodedata

from pydantic import ValidationInfo, field_validator
from utils.nettoyage_donnees import nettoyer_nom_joueur

class PlayerStats(BaseModel):  # Modèle Pydantic
    """Une ligne de joueur de la feuille « Données NBA » (statistiques cumulées sur la saison)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # --- Identité
    player: str = Field(alias="Player", min_length=2)
    team: str = Field(alias="Team", min_length=3, max_length=3)
    age: int = Field(alias="Age", ge=16, le=50)

    # --- Totaux de saison
    gp: int = Field(alias="GP", ge=1, le=82)
    w: int = Field(alias="W", ge=0)
    l: int = Field(alias="L", ge=0)
    pts: int = Field(alias="PTS", ge=0)
    fgm: int = Field(alias="FGM", ge=0)
    fga: int = Field(alias="FGA", ge=0)
    three_pm: int = Field(alias="3PM", ge=0)
    three_pa: int = Field(alias="3PA", ge=0)
    ftm: int = Field(alias="FTM", ge=0)
    fta: int = Field(alias="FTA", ge=0)
    oreb: int = Field(alias="OREB", ge=0)
    dreb: int = Field(alias="DREB", ge=0)
    reb: int = Field(alias="REB", ge=0)
    ast: int = Field(alias="AST", ge=0)
    tov: int = Field(alias="TOV", ge=0)
    stl: int = Field(alias="STL", ge=0)
    blk: int = Field(alias="BLK", ge=0)
    pf: int = Field(alias="PF", ge=0)
    fantasy_pts: int = Field(alias="FP")
    double_doubles: int = Field(alias="DD2", ge=0)
    triple_doubles: int = Field(alias="TD3", ge=0)
    poss: int = Field(alias="POSS", ge=0)

    # --- Moyennes par match
    min_per_game: float = Field(alias="Min", ge=0, le=48)
    plus_minus: float = Field(alias="+/-")

    # --- Pourcentages (0 à 100)
    fg_pct: float = Field(alias="FG%", ge=0, le=100)
    three_pct: float = Field(alias="3P%", ge=0, le=100)
    ft_pct: float = Field(alias="FT%", ge=0, le=100)
    ast_pct: float = Field(alias="AST%", ge=0, le=100)
    oreb_pct: float = Field(alias="OREB%", ge=0, le=100)
    dreb_pct: float = Field(alias="DREB%", ge=0, le=100)
    reb_pct: float = Field(alias="REB%", ge=0, le=100)
    usg_pct: float = Field(alias="USG%", ge=0, le=100)
    efg_pct: float = Field(alias="EFG%", ge=0, le=150)  # > 100 possible sur très peu de tirs
    ts_pct: float = Field(alias="TS%", ge=0, le=150)

    # --- Ratings et ratios
    off_rating: float = Field(alias="OFFRTG", ge=0)
    def_rating: float = Field(alias="DEFRTG", ge=0)
    net_rating: float = Field(alias="NETRTG")
    ast_to_ratio: float = Field(alias="AST/TO", ge=0)
    ast_ratio: float = Field(alias="AST RATIO", ge=0, le=100)
    tov_ratio: float = Field(alias="TO RATIO", ge=0, le=100)
    pace: float = Field(alias="PACE", gt=0)
    pie: float = Field(alias="PIE")  # peut être négatif

    # --- Anomalies non bloquantes, remplies par le validateur (jamais lues dans l'Excel)
    anomalies: list[str] = Field(default_factory=list, exclude=True)

    @field_validator("player", mode="before")
    @classmethod
    def nettoyer_nom(cls, valeur):
        return nettoyer_nom_joueur(valeur)

    @field_validator("team", mode="before")
    @classmethod
    def verifier_equipe(cls, valeur, info: ValidationInfo):
        code = str(valeur).strip().upper()
        codes_connus = (info.context or {}).get("codes_equipes")
        if codes_connus is not None and code not in codes_connus:
            raise ValueError(f"Équipe inconnue : {code!r}")
        return code

    @model_validator(mode="after")
    def regles_metier(self):
        """Règles de cohérence entre plusieurs champs."""
        # 1. Règles bloquantes : une ligne qui les viole est inexploitable
        if self.w + self.l != self.gp:
            raise ValueError(f"W + L ({self.w}+{self.l}) différent de GP ({self.gp})")
        for reussis, tentes, nom in [
            (self.fgm, self.fga, "tirs"),
            (self.three_pm, self.three_pa, "tirs à 3 points"),
            (self.ftm, self.fta, "lancers francs"),
        ]:
            if reussis > tentes:
                raise ValueError(f"{nom} : {reussis} réussis > {tentes} tentés")
        if self.three_pa > self.fga:
            raise ValueError(f"3PA ({self.three_pa}) > FGA ({self.fga})")
        if self.triple_doubles > self.double_doubles:
            raise ValueError(f"TD3 ({self.triple_doubles}) > DD2 ({self.double_doubles})")

        # 2. Règles non bloquantes : on garde la valeur source, on signale l'écart
        if self.oreb + self.dreb != self.reb:
            self.anomalies.append(
                f"REB ({self.reb}) différent de OREB + DREB ({self.oreb}+{self.dreb}={self.oreb + self.dreb})"
            )
        if abs(self.off_rating - self.def_rating - self.net_rating) > 0.2:
            self.anomalies.append(
                f"NETRTG ({self.net_rating}) différent de OFFRTG - DEFRTG ({self.off_rating - self.def_rating:.1f})"
            )
        return self


from typing import Literal

# Valeurs autorisées pour le type de document
DocType = Literal["joueur", "tableau", "texte"]


class RawDocument(BaseModel):
    """Un document extrait par le loader (un joueur, une feuille, un PDF)."""
    page_content: str = Field(min_length=20)      
    source: str = Field(min_length=1)            
    filename: str
    doc_type: DocType


class Chunk(BaseModel):
    """Un morceau de document prêt à être vectorisé."""
    id: str
    text: str = Field(min_length=20, max_length=2500)              
    source: str
    doc_type: DocType


class EmbeddedChunk(BaseModel):
    """Un chunk accompagné de son vecteur d'embedding."""
    chunk: Chunk                        
    embedding: list[float]

    @model_validator(mode="after")
    def verifier_vecteur(self):
        """Vérifie que le vecteur d'embedding est exploitable."""
        # 1. Le vecteur doit avoir exactement la dimension de mistral-embed
        if len(self.embedding) != 1024:
            raise ValueError(
                f"Dimension d'embedding incorrecte : {len(self.embedding)} au lieu de 1024"
            )
        # 2. Le vecteur ne doit pas être entièrement nul (signe d'un échec de l'API)
        if not any(v != 0 for v in self.embedding):
            raise ValueError("Vecteur d'embedding entièrement nul (échec probable de l'API)")
        return self

def _normaliser_source(texte: str) -> str:
    """Rend la comparaison des sources tolérante : crochets, guillemets, casse et espaces ignorés."""
    texte = unicodedata.normalize("NFKC", texte).strip().strip("[]«»\"' ").lower()
    return " ".join(texte.split())


class ReponseAssistant(BaseModel):
    """Sortie structurée imposée au LLM par l'agent Pydantic AI, avec contrôle de cohérence."""

    reponse: str = Field(
        description="Réponse à la question, fondée uniquement sur le contexte, en français."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Sources du contexte réellement utilisées, recopiées EXACTEMENT telles qu'indiquées "
                    "après « Source: » (ex. « regular NBA.xlsx (Joueur: James Harden) »). "
                    "Liste vide si l'information n'est pas disponible.",
    )
    information_disponible: bool = Field(
        description="True si le contexte contient l'information demandée, False sinon "
                    "(information absente des données ou question hors du périmètre NBA).",
    )

    @field_validator("reponse")
    @classmethod
    def reponse_non_vide(cls, valeur: str) -> str:
        if not valeur.strip():
            raise ValueError("La réponse est vide.")
        return valeur.strip()

    @model_validator(mode="after")
    def verifier_coherence(self, info: ValidationInfo):
        # 1. Une réponse affirmative doit être sourcée
        if self.information_disponible and not self.sources:
            raise ValueError("information_disponible=True mais aucune source citée : "
                             "cite au moins une source du contexte.")
        # 2. Un refus ne cite aucune source
        if not self.information_disponible and self.sources:
            raise ValueError("information_disponible=False mais des sources sont citées : "
                             "un refus ne cite aucune source.")
        # 3. Les sources citées doivent exister dans le contexte fourni (si on le connaît)
        autorisees = (info.context or {}).get("sources_autorisees")
        if autorisees is not None:
            valides = {_normaliser_source(s) for s in autorisees}
            inconnues = [s for s in self.sources if _normaliser_source(s) not in valides]
            if inconnues:
                raise ValueError(f"Sources absentes du contexte : {inconnues}. "
                                 f"Sources autorisées : {sorted(autorisees)}")
        return self

class Team(BaseModel):
    """Une ligne de la feuille « Equipe » : code à 3 lettres et nom complet."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    code: str = Field(alias="Code", pattern=r"^[A-Z]{3}$")
    name: str = Field(alias="Nom complet de l'équipe", min_length=3)
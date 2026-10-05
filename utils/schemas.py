# utils/schemas.py — modèles Pydantic : le « contrat » de chaque donnée du pipeline
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlayerStats(BaseModel): #Modèle Pydantic
    """Une ligne de joueur de la feuille « Données NBA » (statistiques cumulées sur la saison)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore") #populate_by_name permet de créer l'objet alias ou avec le nom Python choisi

    player: str = Field(alias="Player", min_length=2) #nom pyhton, complété avec alias qui ramene au nom de la variable sur le tableur d'inputs
    team: str = Field(alias="Team", min_length=3, max_length=3)
    age: int = Field(alias="Age", ge=16, le=50)
    gp: int = Field(alias="GP", ge=1, le=82)
    w: int = Field(alias="W", ge=0)
    l: int = Field(alias="L", ge=0)
    fgm: int = Field(alias="FGM", ge=0)
    fga: int = Field(alias="FGA", ge=0)
    fg_pct: float = Field(alias="FG%", ge=0, le=100)
    three_pm: int = Field(alias="3PM", ge=0)
    three_pa: int = Field(alias="3PA", ge=0)
    three_pct: float = Field(alias="3P%", ge=0, le=100)
    ftm: int = Field(alias="FTM", ge=0)
    fta: int = Field(alias="FTA", ge=0)
    ft_pct: float = Field(alias="FT%", ge=0, le=100)

    @model_validator(mode="after") #applications de quelques règles de logique à respecter pour s'assurer le sens des datas.
    def regles_metier(self):
        """Règles de cohérence entre plusieurs champs."""
        if self.w + self.l != self.gp:
            raise ValueError(f"W + L ({self.w}+{self.l}) différent de GP ({self.gp})") #Si nb matchs gagnés plus perdus différent de nb matchs totaux = on lève une valueerror.
        for reussis, tentes, nom in [
            (self.fgm, self.fga, "tirs"),
            (self.three_pm, self.three_pa, "tirs à 3 points"),
            (self.ftm, self.fta, "lancers francs"),
        ]:
            if reussis > tentes:
                raise ValueError(f"{nom} : {reussis} réussis > {tentes} tentés")
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
    text: str = Field(min_length=20, max_length=2000)              
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
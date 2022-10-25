from pydantic import BaseModel


class FraseGeral(BaseModel):
    frase: str | None = None
    sentimento: float | None = None

    class Config:
        schema_extra = {
            "example": {
                "frase": "Python é ótimo para Machine Learning.",
                "sentimento": 0.1,
            }
        }


class FrasePost(FraseGeral):
    frase: str


class HouseGeral(BaseModel):
    age: float | None = None
    cotacao: float | None = None

    class Config:
        schema_extra = {
            "example": {
                "age": 42,
                "cotacao": 1000.0,
            }
        }


class HousePost(HouseGeral):
    age: float

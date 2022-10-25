from fastapi import FastAPI, status
from textblob import TextBlob

from housing_model import MODEL
from schemas import FraseGeral, FrasePost, HouseGeral, HousePost

app = FastAPI(title="MLops", description="Machine Learning project", version="0.0.1")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post(
    "/sentimento/",
    response_model=FraseGeral,
    status_code=status.HTTP_201_CREATED,
)
async def get_model(frase: FrasePost) -> FraseGeral:
    tb = TextBlob(frase.frase)
    tb_eng = tb.translate(from_lang="pt", to="en")

    dict = frase.dict()
    dict["sentimento"] = tb_eng.sentiment.polarity

    return FraseGeral(**dict)


@app.post(
    "/cotacao/",
    response_model=HouseGeral,
    status_code=status.HTTP_201_CREATED,
)
async def get_model(house: HousePost) -> HouseGeral:
    dict = house.dict()
    dict["cotacao"] = MODEL.predict([[house.age]])

    return HouseGeral(**dict)

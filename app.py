from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI(title="Sentiment Analysis API")

# Загружаем модель и векторизатор
model = joblib.load("artifacts/model.pkl")
vectorizer = joblib.load("artifacts/vectorizer.pkl")

# Входная структура: список текстов
class TextListIn(BaseModel):
    texts: List[str]

# Выходная структура: список результатов
class PredictionOut(BaseModel):
    sentiment: int
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: TextListIn):
    if not data.texts:
        return {"error": "Empty list"}

    # Векторизация
    X = vectorizer.transform(data.texts)

    # Предсказания
    probs = model.predict_proba(X)
    
    results = []
    for p in probs:
        label = int(p.argmax())
        
        score = float(p.max())
        results.append({"sentiment": label, "score": score})
    
    return results
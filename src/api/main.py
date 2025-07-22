from fastapi import FastAPI, Body, HTTPException
import uvicorn
from .schemas import PredictionResponse, PredictionRequest
from models.model_loader import get_model


app = FastAPI(
    title="Предсказание объемов продаж",
    desctiption="Предсказание объемов продаж разных категории товаров на основе Prophet",
    version="0.1",
    contact={
        "name": "Kamran Kurbanov",
        "email": "kamran@kurbanov.me"
    }
)

# Маршрут для проверки работоспособности сервиса
@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "Сервис работает!"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)

@app.post("/api/predict/", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest):
    try:
        feature_list = [request.data]
        print("Получено:", feature_list)

        return {"predictions": [100.0, 110.0, 105.0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибочка вышла: {e}")
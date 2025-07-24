from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from .schemas import PredictionResponse, PredictionRequest
from .validation import validate_data
from src.model.predict import make_predictions
from models.model_loader import get_model
from datetime import datetime, timedelta
import pandas as pd
from fastapi.templating import Jinja2Templates
from src.utils.config import load_config
import os

app = FastAPI(
    title="Предсказание объемов продаж",
    desctiption="Предсказание объемов продаж разных категории товаров на основе Prophet",
    version="0.1",
    contact={
        "name": "Kamran Kurbanov",
        "email": "kamran@kurbanov.me"
    }
)

# Статические файлы и шаблоны
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
templates = Jinja2Templates(directory="src/api/templates")


# Маршрут для проверки работоспособности сервиса
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/docs-ui", response_class=HTMLResponse)
def documentation(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})


@app.get("/predict/excel", responses={
    200: {"description": "Excel файл с прогнозом", "content": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}}}
})
def get_excel_prediction(
    category: list[str] = Query(..., description="Список категорий"),
    date_start: str = Query(..., description="Начало периода прогноза"),
    date_end: str = Query(..., description="Конец периода прогноза")
):  
    config = load_config()
    try:
        start = datetime.strptime(date_start, "%Y-%m-%d")
        end = datetime.strptime(date_end, "%Y-%m-%d")

        if end < start:
            raise HTTPException(status_code=400, detail="Дата окончания не может быть раньше даты начала")

        delta = end - start
        dates = [start + timedelta(days=i) for i in range(delta.days + 1)]

        result_data = []

        for product_name in category:
            predictions = make_predictions(product_name, start, end)

            row = [product_name] + [round(float(p), 0) for p in predictions]
            result_data.append(row)

        # Делаем датафрейм
        columns = ["Номенклатура"] + dates
        df = pd.DataFrame(result_data, columns=columns)



        # Сохраняем в Excel

        output_excel = config["excel_path"]
        os.makedirs(output_excel, exist_ok=True)  

        safe_category = "".join(c for c in category if c.isalnum() or c in " _-").rstrip()
        file_date_start = start.strftime("%Y-%m-%d")
        file_date_end = end.strftime("%Y-%m-%d")
        filename = f"прогноз_{safe_category}_{file_date_start}_до_{file_date_end}.xlsx"
        file_path = os.path.join(output_excel, filename)

        print(f"Сохраняем в: {file_path}")

        df.to_excel(file_path, index=False, engine="openpyxl")
        print("10")

        return FileResponse(
            path=file_path,
            filename=filename,  
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибочка вышла: {e}")
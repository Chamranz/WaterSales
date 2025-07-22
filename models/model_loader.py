import pickle
import joblib
import prophet


# Функиция для загрузки модели для выбранной категории товара
def get_model(product_name: str) -> prophet.forecaster.Prophet:
    return joblib.load(f"models/basement/{product_name}.pkl")

print(type(get_model("00.   ВОДА 19 литров_model")))
from prophet import Prophet
from typing import Tuple, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def train_model(df: pd.DataFrame, product_name: str, random_state=42, test_size=0.2) -> Tuple[Any, Any, Any]:
    """
    Обучает Prophet

    Args:
        X: Признаки
        y: Вектор целевой переменной
        random_state: Сид для воспроизводимости
        test_size: Доля тестовой выборки

    Returns:
        Обученная модель и данные для валидирования
    """

    # Фильруем данные для предикта конкретного класса
    product_data = df[df["Номенклатура"] == product_name].copy()

    # Подготовливаем данные для предикта
    prophet_data = product_data[["ds", "y"]].copy()
    prophet_data = prophet_data.sort_values("ds")

    # Обучаем модель только на train
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_data)

    return model
    

def main_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает данные для обучения модели 

    Args:
        df: Массив с данным со всеми категориями
    
    Returns:
        X: Признаки
        y: Вектор целевой переменной
    """

    # Get all unique products
    all_products = df["Номенклатура"].unique()
    print(f"Generating forecasts for {len(all_products)} products")

    # Create a single large DataFrame to hold all forecasts
    # First, generate all dates for 2024
    dates_2024 = pd.date_range(start=datetime(2024, 1, 1), end=datetime(2024, 12, 31), freq='D')
    date_columns = [date.strftime('%d.%m.%Y') for date in dates_2024]

    # Create empty DataFrame with products as rows and dates as columns
    forecast_matrix = pd.DataFrame(index=all_products, columns=date_columns)
    forecast_matrix.index.name = "Номенклатура"

    # Loop through each product
    for i, product_name in enumerate(all_products):
        try:
            train_model(df, product_name)
        except Exception as e:
            print(f"Error processing product '{product_name}': {e}")

    # Add "Общий итог" (Grand Total) column for each product
    #forecast_matrix["Общий итог"] = forecast_matrix.sum(axis=1)

    # Reset index to make "Номенклатура" a regular column
    #forecast_matrix = forecast_matrix.reset_index()


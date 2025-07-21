import pandas as pd  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
    
def evaluate(model, test: pd.DataFrame, product_name) -> None:
    """
    Оценивает модель на тестовой выборке
    """
    # Прогноз
    future = test[['ds']]  # делаем прогноз только на тестовых датах
    forecast = model.predict(future)

    # Объединяем прогноз и реальные значения
    test_forecast = test.merge(forecast[['ds', 'yhat']], on='ds', how='left')

    # Вычисляем метрики
    test_true = test_forecast['y'].values
    test_pred = test_forecast['yhat'].values

    mae = mean_absolute_error(test_true, test_pred)
    rmse = mean_squared_error(test_true, test_pred)

    print(f"Product: {product_name} | MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # График
    plt.figure(figsize=(12, 4))
    plt.plot(test_forecast['ds'], test_forecast['y'], label='Real')
    plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Predicted')
    plt.title(product_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

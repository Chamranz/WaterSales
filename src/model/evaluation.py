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
    #plt.show()





    # Создаём даты на август 2025 (по дням)
    future_dates = pd.date_range(start='2025-08-01', end='2025-08-31', freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    # Прогноз
    forecast = model.predict(future_df)

    # Выводим основной прогноз
    print(f"Прогноз для {product_name} на август 2025:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

    # График прогноза
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Прогноз', color='blue')
    #plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
    #                color='blue', alpha=0.2, label='Доверительный интервал')
    plt.title(f"Прогноз продаж {product_name} — август 2025")
    plt.xlabel("Дата")
    plt.ylabel("Прогнозируемое значение")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
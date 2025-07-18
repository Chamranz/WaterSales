import pandas as pd  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

    
def evaluate(model, test: pd.DataFrame, all_products: list) -> None:
    """
    Оценивает модель на всех продуктах
    """

    # Берем только нужные колонки
    df = test[["ds"] + list(test.columns[test.columns.str.startswith("product_")]) + ["y"]]

    # Все даты
    all_dates = df['ds'].unique()

    for product in all_products:
        # Создаем future только с нужным продуктом
        future = pd.DataFrame({'ds': all_dates})
        for col in df.columns:
            if col.startswith("product_"):
                future[col] = 1 if col == f"product_{product}" else 0

        # Прогноз
        forecast = model.predict(future)

        # Фильтруем тестовые данные для этого продукта
        product_test = df[df[f"product_{product}"] == 1].copy()
        product_test = product_test.merge(forecast[['ds', 'yhat']], on='ds', how='left')

        # Метрики
        mae = mean_absolute_error(product_test['y'], product_test['yhat'])
        rmse = mean_squared_error(product_test['y'], product_test['yhat'], squared=False)

        print(f"Product: {product} | MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # График
        plt.figure(figsize=(12, 4))
        plt.plot(product_test['ds'], product_test['y'], label='Real')
        plt.plot(product_test['ds'], product_test['yhat'], label='Predicted')
        plt.title(product)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

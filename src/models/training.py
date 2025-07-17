from prophet import Prophet
from typing import Tuple, Any
from sklearn.model_selection import train_test_split

def train_model(X, y, random_state=42, test_size=0.2) -> Tuple[Any, Any, Any]:
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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
        )

    #pd.DataFrame({ds:})

    prophet_data = pd.concat([df, _df])    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model fit(X+)
from prophet import Prophet
from typing import Tuple, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.model.evaluation import evaluate
import pickle
from src.utils.config import load_config
from models import model_loader
import joblib
from models.model_loader import get_model


def make_predictions(product_name, start, end) -> List:
    model = get_model(product_name)

    future_dates = pd.date_range(start=start, end=end, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    # Прогноз
    forecast = model.predict(future_df)
    return forecast["yhat"].tolist()
 

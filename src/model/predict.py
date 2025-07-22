from prophet import Prophet
from typing import Tuple, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.models.evaluation import evaluate
import pickle
from src.utils.config import load_config
from models import model_loader
import joblib
from models.model_loader import get_model


def predict_sales(product_name: str, period: dict) -> List:
    model = 
 

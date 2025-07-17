import pandas as pd
from src.utils.config import load_config, get_data_paths
import numpy as np
from datetime import datetime
from prophet import Prophet
import os


def preprocess_data(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет данные за 5 лет, соединяет все в один список

    Args:
        _df: исходный DataFrame

    Returns:
        Очищенный DataFrame, где признаковое пространство слито в один список
    """

    years = ["2019", "2020", "2021", "2022", "2023"]
    df = None
    for year in years:
        if df is None:
            df = _df
            continue
        df = pd.concat([df, _df])

    # конкатенация
    df = df.melt(
        id_vars=["Номенклатура"],
        var_name="ds",
        value_name="y")

    return df
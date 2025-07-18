import pandas as pd
from src.utils.config import load_config, get_data_paths
import numpy as np
from datetime import datetime
from prophet import Prophet
import os
from scipy.stats import zscore


def remove_outliers_iqr(group):
    Q1 = group['y'].quantile(0.25)
    Q3 = group['y'].quantile(0.75)
    IQR = Q3 - Q1
    return group[(group['y'] >= (Q1 - 1.5 * IQR)) & (group['y'] <= (Q3 + 1.5 * IQR))]


def preprocess_data(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Переводит данные из широкого формата (даты как столбцы) в длинный (даты как строки)

    Args:
        _df: исходный DataFrame

    Returns:
        Очищенный DataFrame, где признаковое пространство слито в один список
        Списко всех уникальных продуктов
    """

    # Конкатенация со всех лет
    #years = ["2019", "2020", "2021", "2022", "2023"]
    years = ["2019"]
    df = None
    for year in years:
        if df is None:
            df = _df
            continue
        df = pd.concat([df, _df])

    # Преобразование в один длинный список
    df = df.melt(
        id_vars=["Номенклатура"],
        var_name="ds",
        value_name="y")

    # Сбор всех классов
    all_products = df["Номенклатура"].unique()

    # Фильтрация лишних строк
    df = df[~df["ds"].str.contains("Общий итог", na=False)]
    df = df[~df["Номенклатура"].str.contains("ИТОГО:", na=False)]

    # Преобразует столбец ds в тип datatime, переводит таргеты во float
    df["ds"] = pd.to_datetime(df["ds"], format="%d.%m.%Y")
    df["y"] = df["y"].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)

    # Заполнение пропщенных значении на основе средних значений по тому же дню в году
    df['month_day'] = df['ds'].dt.strftime('%m-%d')
    mean_values = df.groupby(['Номенклатура', 'month_day'])['y'].transform('mean')
    df['y'] = df['y'].fillna(mean_values)

    # Потом заполняем сердним по продукту и то что осталось - нулями
    product_means = df.groupby('Номенклатура')['y'].transform('mean')
    df['y'] = df['y'].fillna(product_means)
    df['y'] = df['y'].fillna(0)

    # Чистим выбрросы
    df_long = df[df["y"] > 0]

    df_long['z_score'] = df_long.groupby("Номенклатура")["y"].transform(lambda x: zscore(x, nan_policy='omit'))
    df_clean = df_long[df_long['z_score'].abs() < 3]  # оставляем только те, у которых z-score < 3
    
    return df_clean, all_products
    

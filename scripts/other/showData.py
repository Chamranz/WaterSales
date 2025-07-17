import pandas as pd
from src.utils.config import load_config, get_data_paths
import numpy as np
from datetime import datetime
from prophet import Prophet
import os

config = load_config()
data_path, output_path = get_data_paths(config)
df = pd.read_excel(data_path) 

# предобработка
years = ["2019", "2020", "2021", "2022", "2023"]
df = None
for year in years:
    _df = pd.read_excel(data_path, sheet_name=year)
    if df is None:
        df = _df
        continue
    df = pd.concat([df, _df])

# конкатенация
df = df.melt(
    id_vars=["Номенклатура"],
    var_name="ds",
    value_name="y")

print(df.head)
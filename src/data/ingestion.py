import pandas as pd
from typing import Tuple

def load_data(data_path: str) -> pd.DataFrame: 
    """
        Загрузка данных

        Args:
            data_path: Путь к файлу с данными

        Returns:
            DataFrame с даггыми
    """

    years = ["2019", "2020", "2021", "2022", "2023"]
    df = None
    for year in years:
        _df = pd.read_excel(data_path, sheet_name=year)
        if df is None:
            df = _df
            continue
        df = pd.concat([df, _df])
    return df
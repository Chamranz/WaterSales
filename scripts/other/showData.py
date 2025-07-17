import pandas as pd
from src.utils.config import load_config, get_data_paths
from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
import numpy as np
from datetime import datetime
from prophet import Prophet
import os

config = load_config()
data_path, output_path = get_data_paths(config)

data = load_data(data_path)
df = preprocess_data(data)

print(df.head)
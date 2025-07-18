import pandas as pd
from src.utils.config import load_config, get_data_paths
from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
import numpy as np
from datetime import datetime
from prophet import Prophet
import os
import matplotlib.pyplot as plt

config = load_config()
data_path, output_path = get_data_paths(config)

data = load_data(data_path)
df = preprocess_data(data)

plt.plot(df["ds"], df["y"])
#plt.show()

print(df.head)
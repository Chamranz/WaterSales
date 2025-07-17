import pandas as pd
from src.utils.config import load_config, get_data_paths

config = load_config()
data_path, output_path = get_data_paths(config)
df = pd.read_excel(data_path) 
print(df.head)
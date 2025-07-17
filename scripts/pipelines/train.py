from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.utils.config import load_config, get_data_paths

config = load_config()
data_path, output_path = get_data_paths(config)

data = load_data(data_path)
data_clear = preprocess_data(data)

print(data_clear.head)


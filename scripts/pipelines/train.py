from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.utils.config import load_config, get_data_paths
from src.models.training import main_training, train_model
from src.models.evaluation import evaluate
import pickle

def main() -> None:
    config = load_config()
    data_path, output_path = get_data_paths(config)

    data = load_data(data_path)
    data_clear = preprocess_data(data)

    model = main_training(data_clear)

    #evaluate(model, test, all_products)

if __name__ == "__main__":
    main()




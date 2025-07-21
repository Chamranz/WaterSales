from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.utils.config import load_config, get_data_paths
from src.models.training import main_training
from src.models.evaluation import evaluate
import pickle

def main() -> None:
    config = load_config()
    data_path, output_path = get_data_paths(config)

    data = load_data(data_path)
    data_clear = preprocess_data(data)

    model, test = main_training(data_clear)
    evaluate(model)

    with open(config["output_path"], "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()




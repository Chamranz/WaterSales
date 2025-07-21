import json
from pathlib import Path
from typing import Tuple

def load_config(config_path: Path = Path("config.json")) -> dict:
    """
    Загружает конфигурационные параметры из JSON-файла

    Args:
        config_path: Путь к конфигурационному файлу
    
    Returns:
        Словарь с параметрами конфигурации.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def get_data_paths(config: dict) -> Tuple[Path, Path]:
    """
    Извлекает пути к данным модели из конфигурации

    Args:
        config: Словарь с параметрами конфигурации
    
    Returns:
        Кортеж из двух путей: путь к данным и путь для сохранеиня модели
    """
    return Path(config["data_path"]), Path(config["output_path"])
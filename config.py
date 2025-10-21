import json
import os

# Define the path to your config file
CONFIG_FILE_PATH = "config.json"  # Adjust path if config is in a subdirectory


def load_config(config_path: str = CONFIG_FILE_PATH) -> dict:
    """
    Loads configuration from a JSON file.

    Args:
        config_path: Path to the config.json file.

    Returns:
        A dictionary containing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON configuration file {config_path}: {e}")


# Load the configuration once when the module is imported or script starts
CONFIG = load_config()

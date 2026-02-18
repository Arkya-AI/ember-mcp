import json
from pathlib import Path

from ember.models import EmberConfig

# Base directory constants
EMBER_DIR = Path.home() / ".ember"
CONFIG_FILE = EMBER_DIR / "config.json"

# Subdirectories
DIR_EMBERS = EMBER_DIR / "embers"
DIR_INDEX = EMBER_DIR / "index"
DIR_CELLS = EMBER_DIR / "cells"

DEFAULT_CONFIG_DICT = {
    "k_cells": 16,
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "similarity_threshold": 0.4,
    "data_dir": str(EMBER_DIR),
}


def _ensure_directories(base_path: Path) -> None:
    """Ensures the base directory and required subdirectories exist."""
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "embers").mkdir(exist_ok=True)
    (base_path / "index").mkdir(exist_ok=True)
    (base_path / "cells").mkdir(exist_ok=True)


def load_config() -> EmberConfig:
    """
    Loads the configuration from config.json.
    If the file or directories do not exist, they are created with defaults.
    """
    if not EMBER_DIR.exists():
        _ensure_directories(EMBER_DIR)

    if not CONFIG_FILE.exists():
        config = EmberConfig()
        save_config(config)
        return config

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "data_dir" in data and isinstance(data["data_dir"], str):
            data["data_dir"] = Path(data["data_dir"])

        config = EmberConfig(**data)
        _ensure_directories(config.data_dir)
        return config

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}. Reverting to defaults.")
        config = EmberConfig()
        save_config(config)
        return config


def save_config(config: EmberConfig) -> None:
    """Persists the current configuration to config.json."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ensure_directories(config.data_dir)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=2))

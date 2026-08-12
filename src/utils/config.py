"""Configuration loader."""

from pathlib import Path
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/settings.yaml") -> dict:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return yaml.safe_load(f)

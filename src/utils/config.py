"""Configuration loader."""

from pathlib import Path
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/settings.yaml") -> dict:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # encoding= explicit (2026-09-01, full-codebase review) -- this is the config loader
    # for the whole system, and settings.yaml is real UTF-8 (it already contains "§").
    # A bare open() decodes with the locale codec, so on the supported Windows dev box
    # it mojibakes that content today and would raise UnicodeDecodeError at boot the
    # moment the file gains a character whose UTF-8 bytes include one of cp1252's
    # undefined bytes (a curly quote suffices) -- the app failing to start on Windows
    # while booting fine on the Linux production box. CLAUDE.md's encoding rule covers
    # src/**, and bare open() is the variant its AST test cannot see.
    return yaml.safe_load(path.read_text(encoding="utf-8"))

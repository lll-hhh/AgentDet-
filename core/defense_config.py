from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def defense_config_path(filename: str) -> Path:
    return CONFIG_DIR / filename


def load_json_config(filename: str, default: Any = None) -> Any:
    path = defense_config_path(filename)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

"""Load key=value pairs from a local .env file into process env (no extra deps)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    if path is None:
        # project root = parent of src/
        path = Path(__file__).resolve().parents[1] / ".env"
    path = Path(path)
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Do not overwrite an already-set environment variable
        if key and key not in os.environ:
            os.environ[key] = value
    return path

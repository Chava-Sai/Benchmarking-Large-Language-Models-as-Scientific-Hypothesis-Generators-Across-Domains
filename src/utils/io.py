"""Shared I/O helpers."""

import json
from pathlib import Path


def load_jsonl(path: Path | str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data: list[dict], path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_json(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path | str, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

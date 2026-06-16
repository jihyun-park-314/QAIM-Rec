"""Shared utilities for src/pilot/ scripts (plan.md §1, §4 Stage P1-P4)."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str, cls: type[T]) -> T:
    """Load a YAML file and construct a config dataclass."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in fields}
    return cls(**filtered)


def write_report(report: object, path: str) -> None:
    """Serialize a Report dataclass to JSON at path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, indent=2, ensure_ascii=False)

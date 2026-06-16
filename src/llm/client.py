"""M1: Ollama LLM client with SQLite cache and latency tracking.

All LLM calls go through LLMClient.generate().
Cache key = sha256(prompt_version | model_id | temperature | prompt).
Cache hit returns immediately (latency_s=0.0 recorded separately).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import requests
import yaml


@dataclass
class LLMConfig:
    model_id: str
    api_url: str
    temperature: float
    max_new_tokens: int
    prompt_version: str
    cache_db: str
    retry_max: int = 2
    self_consistency_t: Optional[int] = None


def load_llm_config(path: str) -> LLMConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return LLMConfig(
        model_id=data["model_id"],
        api_url=data["api_url"],
        temperature=float(data.get("temperature", 0.0)),
        max_new_tokens=int(data.get("max_new_tokens", 512)),
        prompt_version=data.get("prompt_version", "v1"),
        cache_db=data.get("cache_db", "data/cache/llm_cache.sqlite"),
        retry_max=int(data.get("retry_max", 2)),
        self_consistency_t=data.get("self_consistency_t"),
    )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        os.makedirs(os.path.dirname(os.path.abspath(config.cache_db)), exist_ok=True)
        self._conn = sqlite3.connect(config.cache_db, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, response TEXT)"
        )
        self._conn.commit()

    def _cache_key(self, prompt: str) -> str:
        c = self.config
        raw = f"{c.prompt_version}|{c.model_id}|{c.temperature}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, prompt: str) -> tuple[str, float, bool]:
        """Return (response_text, latency_s, cache_hit).

        latency_s is 0.0 on cache hit (call was free).
        """
        key = self._cache_key(prompt)
        row = self._conn.execute(
            "SELECT response FROM cache WHERE key=?", (key,)
        ).fetchone()
        if row:
            return row[0], 0.0, True

        c = self.config
        payload = {
            "model": c.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": c.temperature,
                "num_predict": c.max_new_tokens,
            },
        }
        t0 = time.perf_counter()
        resp = requests.post(c.api_url, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["response"]
        latency = time.perf_counter() - t0

        self._conn.execute(
            "INSERT OR IGNORE INTO cache (key, response) VALUES (?, ?)",
            (key, text),
        )
        self._conn.commit()
        return text, latency, False

    def close(self) -> None:
        self._conn.close()

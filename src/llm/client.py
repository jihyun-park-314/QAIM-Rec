"""M1: Ollama LLM client with two-tier SQLite cache and latency tracking.

All LLM calls go through LLMClient.generate().

Cache design (two-tier):
  parsed_success_cache — key=(prompt_version|model_id|temperature|messages) [no max_new_tokens]
      Stores parsed JSON str.  Written only on successful parse.  Read first on every call.
      Hit → (stored_json_str, "stop", 0.0, True).  Empty responses NEVER stored here.
  raw_debug_cache — key=(prompt_version|model_id|temperature|max_new_tokens|messages)
      Stores raw Ollama response + done_reason for debugging.  Non-empty responses only.
      Hit → (raw_text, stored_done_reason, 0.0, True).

Uses /api/chat with messages format — required for chat-tuned models (gemma4:26b).
Sends think:false as a top-level field (not inside options) per Ollama /api/chat spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
import yaml


@dataclass
class LLMConfig:
    model_id: str
    api_url: str
    temperature: float
    max_new_tokens: int          # default budget per task (e.g. 1200 for p1_base)
    prompt_version: str
    cache_db: str
    retry_max: int = 2
    retry_max_new_tokens: int = 4096   # budget used on retry
    think: bool = False                 # Ollama think option (False = disable thinking tokens)
    format_json: bool = True            # Ollama format:"json" option
    self_consistency_t: Optional[int] = None


def load_llm_config(path: str) -> LLMConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    fields = {f.name for f in dataclasses.fields(LLMConfig)}
    filtered = {k: v for k, v in data.items() if k in fields}
    # Allow runtime override of api_url (useful in Docker where Ollama is on host)
    api_url = os.environ.get("OLLAMA_API_URL", data["api_url"])
    return LLMConfig(
        model_id=data["model_id"],
        api_url=api_url,
        temperature=float(data.get("temperature", 0.0)),
        max_new_tokens=int(data.get("max_new_tokens", 512)),
        prompt_version=data.get("prompt_version", "v1"),
        cache_db=data.get("cache_db", "data/cache/llm_cache.sqlite"),
        retry_max=int(data.get("retry_max", 2)),
        retry_max_new_tokens=int(data.get("retry_max_new_tokens", 4096)),
        think=bool(data.get("think", False)),
        format_json=bool(data.get("format_json", True)),
        self_consistency_t=data.get("self_consistency_t"),
    )


@dataclass
class P1CallResult:
    """Per-sample result from a p1_base / p1_aspect LLM call (with retry stats)."""
    parsed: dict | None
    latency_s: float
    cache_hit: bool
    final_max_new_tokens: int
    done_reason: str
    retry_count: int              # 0 = first budget succeeded; 1 = retry used
    empty_response_count: int     # number of empty-string responses from Ollama
    done_reason_length_count: int # number of done_reason=="length" events
    parse_failure_count: int      # json.loads failures
    schema_failure_count: int     # schema / content validation failures
    truncated_input: bool = False # review_text was truncated before sending


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        os.makedirs(os.path.dirname(os.path.abspath(config.cache_db)), exist_ok=True)
        self._conn = sqlite3.connect(config.cache_db, check_same_thread=False)
        # WAL mode: safe for concurrent multi-process writes; idempotent on already-WAL DBs.
        # busy_timeout: retry up to 30s on SQLITE_BUSY instead of raising immediately.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS parsed_success_cache
                (key TEXT PRIMARY KEY, response TEXT);
            CREATE TABLE IF NOT EXISTS raw_debug_cache
                (key TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE IF NOT EXISTS cache
                (key TEXT PRIMARY KEY, response TEXT);
            """
        )
        # Remove any poisoned empty entries that may have accumulated in legacy table.
        self._conn.execute("DELETE FROM cache WHERE response = ''")
        self._conn.commit()

    # ------------------------------------------------------------------ keys

    def _parsed_key(self, messages: list[dict]) -> str:
        """Cache key for parsed_success_cache — excludes max_new_tokens."""
        c = self.config
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        raw = f"{c.prompt_version}|{c.model_id}|{c.temperature}|{serialized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _raw_key(self, messages: list[dict], max_new_tokens: int) -> str:
        """Cache key for raw_debug_cache — includes max_new_tokens."""
        c = self.config
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        raw = f"{c.prompt_version}|{c.model_id}|{c.temperature}|{max_new_tokens}|{serialized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ public API

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        json_schema: dict | None = None,
    ) -> tuple[str, str, float, bool]:
        """Return (raw_text, done_reason, latency_s, cache_hit).

        json_schema: if provided, passed as payload["format"] (Ollama constrained
          decoding).  Overrides the config-level format_json flag.

        Lookup order:
          1. parsed_success_cache (by parsed_key, no max_new_tokens) →
             returns stored_json_str / done_reason="stop" / latency=0 / hit=True
          2. raw_debug_cache (by raw_key, includes max_new_tokens) →
             returns raw_text / stored_done_reason / latency=0 / hit=True
          3. Call Ollama → store non-empty raw in raw_debug_cache → return

        Empty responses are NEVER stored.  Call store_parsed() after successful parse.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        parsed_key = self._parsed_key(messages)
        row = self._conn.execute(
            "SELECT response FROM parsed_success_cache WHERE key=?", (parsed_key,)
        ).fetchone()
        if row:
            return row[0], "stop", 0.0, True

        raw_key = self._raw_key(messages, max_new_tokens)
        row = self._conn.execute(
            "SELECT payload FROM raw_debug_cache WHERE key=?", (raw_key,)
        ).fetchone()
        if row:
            stored = json.loads(row[0])
            return stored["text"], stored["done_reason"], 0.0, True

        c = self.config
        payload: dict = {
            "model": c.model_id,
            "messages": messages,
            "stream": False,
            "think": c.think,
            "options": {
                "temperature": c.temperature,
                "num_predict": max_new_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        elif c.format_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        resp = requests.post(c.api_url, json=payload, timeout=600)
        resp.raise_for_status()
        body = resp.json()
        text: str = body["message"]["content"]
        done_reason: str = body.get("done_reason", "unknown")
        latency = time.perf_counter() - t0

        if text:  # never cache empty responses
            self._conn.execute(
                "INSERT OR IGNORE INTO raw_debug_cache (key, payload) VALUES (?, ?)",
                (raw_key, json.dumps({"done_reason": done_reason, "text": text})),
            )
            self._conn.commit()

        return text, done_reason, latency, False

    def store_parsed(self, messages: list[dict], json_str: str) -> None:
        """Persist a successfully parsed JSON string to parsed_success_cache.

        Only call this after parse_response() has validated the result.
        Empty strings are silently ignored.
        """
        if not json_str:
            return
        key = self._parsed_key(messages)
        self._conn.execute(
            "INSERT OR REPLACE INTO parsed_success_cache (key, response) VALUES (?, ?)",
            (key, json_str),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

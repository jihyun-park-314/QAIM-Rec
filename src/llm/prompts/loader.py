"""Prompt loader: reads .txt files from config/prompts/{dataset}/{domain}/.

Falls back to _default/ if domain-specific file is absent.
Returns (content, version_hash) — version_hash is the first 8 chars of the
sha256 of the file content and is used as prompt_template_version in the
LLM cache key.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

_PROMPTS_BASE = os.path.join(
    os.path.dirname(__file__),  # src/llm/prompts/
    "..", "..", "..",            # project root
    "config", "prompts",
)
_PROMPTS_BASE = os.path.normpath(_PROMPTS_BASE)


@lru_cache(maxsize=None)
def load_prompt(dataset: str, domain: str, prompt_name: str) -> tuple[str, str]:
    """Return (content, version_hash_8).

    Searches domain-specific file first, then _default/.
    Raises FileNotFoundError if neither exists.
    """
    for d in ([domain] if domain != "_default" else []) + ["_default"]:
        path = os.path.join(_PROMPTS_BASE, dataset, d, f"{prompt_name}.txt")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            version = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
            return content, version
    raise FileNotFoundError(
        f"Prompt '{prompt_name}' not found for dataset='{dataset}' "
        f"domain='{domain}' (also checked _default). "
        f"Expected file at: {os.path.join(_PROMPTS_BASE, dataset, '_default', prompt_name + '.txt')}"
    )

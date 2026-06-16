"""P1-base prompt: purpose-centric intent extraction (plan.md §5 P1, narrow baseline).

Prompt content lives in config/prompts/amazon/_default/p1_base.txt.
PROMPT_VERSION is the sha256[:8] of that file — used in the LLM cache key.
"""

from __future__ import annotations

import json
import re

from src.llm.prompts.loader import load_prompt

SYSTEM_PROMPT, PROMPT_VERSION = load_prompt("amazon", "_default", "p1_base")

USER_TEMPLATE = """\
Item title: {title}
Category: {category}
Brand: {brand}
Price: {price}
Rating: {rating}/5
Review: \"\"\"{review_text}\"\"\""""


def build_prompt(item: dict) -> str:
    user_block = USER_TEMPLATE.format(
        title=item.get("title", ""),
        category=item.get("category", ""),
        brand=item.get("brand", ""),
        price=item.get("price", ""),
        rating=item.get("rating", ""),
        review_text=item.get("review_text", ""),
    )
    return f"{SYSTEM_PROMPT}\n\n{user_block}"


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1)
    return text


def parse_response(raw: str, item: dict | None = None) -> dict | None:
    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        return None

    required = {"purpose", "is_discriminative", "disposition_note", "preference_attrs"}
    if not required.issubset(data):
        return None

    purpose = (data.get("purpose") or "").strip()
    if not purpose:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", purpose)
    if len(sentences) > 2:
        return None
    if item:
        item_title = (item.get("title") or "").lower().strip()
        if item_title and purpose.lower().strip() == item_title:
            return None

    return data


def run_p1(client, item: dict, retry_max: int = 2) -> tuple[dict | None, float, bool]:
    """Run p1_base for one item.  Returns (result, latency_s, cache_hit)."""
    prompt = build_prompt(item)
    for attempt in range(retry_max + 1):
        raw, latency, hit = client.generate(prompt)
        parsed = parse_response(raw, item)
        if parsed is not None:
            return parsed, latency, hit
    return None, latency, hit

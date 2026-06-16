"""P1-aspect prompt: domain-general contextual preference extraction (plan.md §5, A/B).

Framing: intent = "query-activatable contextual preference unit" — the specific slice
of a user's preferences that the current interaction can activate.  Purchase purpose
is one possible aspect, not the whole definition.

Prompt content lives in config/prompts/amazon/_default/p1_aspect.txt.
PROMPT_VERSION is the sha256[:8] of that file — used in the LLM cache key.
"""

from __future__ import annotations

import json
import re

from src.llm.prompts.loader import load_prompt

SYSTEM_PROMPT, PROMPT_VERSION = load_prompt("amazon", "_default", "p1_aspect")

USER_TEMPLATE = """\
Item title: {title}
Category: {category}
Brand/Creator: {brand}
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


def _aspect_coverage_valid(ac: dict) -> bool:
    """True if >= 2 of the 4 aspect_coverage fields are non-null/non-empty."""
    count = 0
    for key in ("usage_context", "taste_or_style_preference",
                "selection_criteria", "preference_tradeoff"):
        v = ac.get(key)
        if v is not None and v != "" and v != []:
            count += 1
    return count >= 2


def parse_response(raw: str) -> dict | None:
    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        return None

    required = {
        "contextual_intent", "is_discriminative",
        "aspect_coverage", "disposition_note", "preference_attrs",
    }
    if not required.issubset(data):
        return None

    ci = data.get("contextual_intent")
    if not isinstance(ci, list) or not ci or not ci[0]:
        return None
    if len(ci) > 2:
        data["contextual_intent"] = ci[:2]

    ac = data.get("aspect_coverage") or {}
    data["_aspect_coverage_valid"] = _aspect_coverage_valid(ac)

    return data


def run_p1_aspect(
    client, item: dict, retry_max: int = 2
) -> tuple[dict | None, float, bool]:
    """Run p1_aspect for one item.  Returns (result, latency_s, cache_hit)."""
    prompt = build_prompt(item)
    latency, hit = 0.0, False
    for attempt in range(retry_max + 1):
        raw, latency, hit = client.generate(prompt)
        parsed = parse_response(raw)
        if parsed is not None:
            return parsed, latency, hit
    return None, latency, hit

"""P1-base prompt: purpose-centric intent extraction (plan.md §5 P1, narrow baseline).

Prompt content lives in config/prompts/amazon/_default/p1_base.txt.
PROMPT_VERSION is the sha256[:8] of that file — used in the LLM cache key.

Token budget:
  First attempt : P1_BASE_DEFAULT_TOKENS (1200)
  Retry         : P1_RETRY_TOKENS (4096) — only if first attempt fails.

Failure conditions that trigger retry (plan.md stabilisation spec):
  - empty message.content
  - JSON parse error
  - schema / content validation failure
  - done_reason == "length"
"""

from __future__ import annotations

import json
import re

from src.llm.client import LLMClient, P1CallResult
from src.llm.prompts.loader import load_prompt

SYSTEM_PROMPT, PROMPT_VERSION = load_prompt("amazon", "_default", "p1_base")

P1_BASE_DEFAULT_TOKENS = 1200
P1_RETRY_TOKENS = 4096

_REVIEW_TEXT_MAX_CHARS = 6000

# Passed to Ollama as format parameter for constrained decoding (forces correct field names).
P1_BASE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "purpose": {"type": "string", "maxLength": 200},
        "is_discriminative": {"type": "boolean"},
        "disposition_note": {"type": ["string", "null"], "maxLength": 140},
        "preference_attrs": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "price_band": {
                    "type": "string",
                    "enum": ["budget", "mid-range", "premium", "unknown"],
                },
                "feature_priorities": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {"type": "string", "maxLength": 60},
                },
                "brand_tendency": {"type": "string", "maxLength": 80},
                "style": {"type": ["string", "null"], "maxLength": 80},
                "avoid": {
                    "anyOf": [
                        {
                            "type": "array",
                            "maxItems": 3,
                            "items": {"type": "string", "maxLength": 60},
                        },
                        {"type": "null"},
                    ]
                },
            },
            "required": [
                "price_band", "feature_priorities", "brand_tendency", "style", "avoid"
            ],
        },
    },
    "required": ["purpose", "is_discriminative", "disposition_note", "preference_attrs"],
}

USER_TEMPLATE = """\
Item title: {title}
Category: {category}
Brand: {brand}
Price: {price}
Rating: {rating}/5
Review: \"\"\"{review_text}\"\"\""""


def _truncate_review(text: str) -> tuple[str, bool]:
    """Truncate extremely long reviews: keep first 4000 + last 2000 chars."""
    if len(text) <= _REVIEW_TEXT_MAX_CHARS:
        return text, False
    return text[:4000] + text[-2000:], True


def build_messages(item: dict) -> tuple[list[dict], bool]:
    """Return (messages, was_truncated).  Raw item is never mutated."""
    review_text, truncated = _truncate_review(item.get("review_text", ""))
    user_block = USER_TEMPLATE.format(
        title=item.get("title", ""),
        category=item.get("category", ""),
        brand=item.get("brand", ""),
        price=item.get("price", ""),
        rating=item.get("rating", ""),
        review_text=review_text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ], truncated


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1)
    return text


def parse_response(
    raw: str, item: dict | None = None
) -> tuple[dict | None, str | None]:
    """Validate and parse a P1-base LLM response.

    Returns (parsed_dict, failure_type).
    failure_type is None on success; otherwise one of:
      "json_parse_error" | "schema_error"
    """
    if not raw:
        return None, "schema_error"

    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        return None, "json_parse_error"

    # Normalize "null" strings → None (constrained decoding artefact for string|null fields)
    for k in ("disposition_note",):
        if isinstance(data.get(k), str) and data[k].lower() == "null":
            data[k] = None

    required = {"purpose", "is_discriminative", "disposition_note", "preference_attrs"}
    if not required.issubset(data):
        return None, "schema_error"

    purpose = (data.get("purpose") or "").strip()
    if not purpose:
        return None, "schema_error"
    sentences = re.split(r"(?<=[.!?])\s+", purpose)
    if len(sentences) > 2:
        return None, "schema_error"
    if item:
        item_title = (item.get("title") or "").lower().strip()
        if item_title and purpose.lower().strip() == item_title:
            return None, "schema_error"

    return data, None


def run_p1(
    client: LLMClient, item: dict, retry_max: int = 2
) -> P1CallResult:
    """Run p1_base for one item with two-budget retry logic.

    Budget sequence: [P1_BASE_DEFAULT_TOKENS, P1_RETRY_TOKENS]
    Any failure (empty / parse error / schema error / done_reason=="length")
    triggers a retry with the larger budget.
    Only stores to parsed_success_cache on clean success.
    """
    messages, truncated = build_messages(item)
    budgets = [P1_BASE_DEFAULT_TOKENS]
    if retry_max >= 1:
        budgets.append(P1_RETRY_TOKENS)

    latency_s = 0.0
    cache_hit = False
    done_reason = "unknown"
    final_tokens = budgets[0]
    empty_count = 0
    length_count = 0
    parse_count = 0
    schema_count = 0

    for attempt, max_tokens in enumerate(budgets):
        raw, done_reason, latency_s, cache_hit = client.generate(
            messages, max_new_tokens=max_tokens, json_schema=P1_BASE_JSON_SCHEMA
        )
        final_tokens = max_tokens

        if not raw:
            empty_count += 1
            continue  # retry with next budget

        if done_reason == "length":
            length_count += 1

        parsed, failure = parse_response(raw, item)

        if parsed is not None and done_reason != "length":
            client.store_parsed(messages, json.dumps(parsed, ensure_ascii=False))
            return P1CallResult(
                parsed=parsed,
                latency_s=latency_s,
                cache_hit=cache_hit,
                final_max_new_tokens=final_tokens,
                done_reason=done_reason,
                retry_count=attempt,
                empty_response_count=empty_count,
                done_reason_length_count=length_count,
                parse_failure_count=parse_count,
                schema_failure_count=schema_count,
                truncated_input=truncated,
            )

        if failure == "json_parse_error":
            parse_count += 1
        elif failure == "schema_error":
            schema_count += 1

    return P1CallResult(
        parsed=None,
        latency_s=latency_s,
        cache_hit=cache_hit,
        final_max_new_tokens=final_tokens,
        done_reason=done_reason,
        retry_count=len(budgets) - 1,
        empty_response_count=empty_count,
        done_reason_length_count=length_count,
        parse_failure_count=parse_count,
        schema_failure_count=schema_count,
        truncated_input=truncated,
    )

"""P1-aspect prompt: domain-general contextual preference extraction (plan.md §5, A/B).

Framing: intent = "query-activatable contextual preference unit" — the specific slice
of a user's preferences that the current interaction can activate.  Purchase purpose
is one possible aspect, not the whole definition.

Prompt content lives in config/prompts/amazon/_default/p1_aspect.txt.
PROMPT_VERSION is the sha256[:8] of that file — used in the LLM cache key.

Token budget:
  First attempt : P1_ASPECT_DEFAULT_TOKENS (1600)
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

SYSTEM_PROMPT, PROMPT_VERSION = load_prompt("amazon", "_default", "p1_aspect")

P1_ASPECT_DEFAULT_TOKENS = 1600
P1_RETRY_TOKENS = 4096

_REVIEW_TEXT_MAX_CHARS = 6000

# Passed to Ollama as format parameter for constrained decoding (forces correct field names).
P1_ASPECT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "contextual_intent": {
            "type": "array",
            "minItems": 0,
            "maxItems": 2,
            "items": {"type": "string", "maxLength": 140},
        },
        "is_discriminative": {"type": "boolean"},
        "aspect_coverage": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "usage_context": {"type": ["string", "null"], "maxLength": 100},
                "taste_or_style_preference": {"type": ["string", "null"], "maxLength": 100},
                "selection_criteria": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {"type": "string", "maxLength": 60},
                },
                "preference_tradeoff": {"type": ["string", "null"], "maxLength": 120},
            },
            "required": [
                "usage_context", "taste_or_style_preference",
                "selection_criteria", "preference_tradeoff"
            ],
        },
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
                "brand_or_creator_tendency": {
                    "type": "string",
                    "enum": ["brand", "author", "creator", "designer", "agnostic"],
                },
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
                "price_band", "feature_priorities", "brand_or_creator_tendency",
                "style", "avoid"
            ],
        },
    },
    "required": [
        "contextual_intent", "is_discriminative", "aspect_coverage",
        "disposition_note", "preference_attrs"
    ],
}

USER_TEMPLATE = """\
Item title: {title}
Category: {category}
Brand/Creator: {brand}
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


_NULL_PLACEHOLDERS = frozenset([
    "null", "none", "n/a", "na", "n.a.", "no preference",
    "no signal", "no intent", "no meaningful preference",
])


def _is_null_placeholder(s: str) -> bool:
    return s.strip().lower() in _NULL_PLACEHOLDERS


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1)
    return text


def _coerce_null_strings(d: dict) -> dict:
    """Replace "null" string values with Python None (constrained decoding artefact)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str) and v.lower() == "null":
            result[k] = None
        elif isinstance(v, dict):
            result[k] = _coerce_null_strings(v)
        elif isinstance(v, list):
            result[k] = [
                (None if (isinstance(x, str) and x.lower() == "null") else x)
                for x in v
                if not (isinstance(x, str) and x.lower() == "null")  # remove "null" list items
            ]
        else:
            result[k] = v
    return result


def _aspect_coverage_valid(ac: dict) -> bool:
    """True if >= 2 of the 4 aspect_coverage fields are non-null/non-empty."""
    count = 0
    for key in ("usage_context", "taste_or_style_preference",
                "selection_criteria", "preference_tradeoff"):
        v = ac.get(key)
        if v is not None and v != "" and v != []:
            count += 1
    return count >= 2


def parse_response(raw: str) -> tuple[dict | None, str | None]:
    """Validate and parse a P1-aspect LLM response.

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

    # Check for null-placeholders in contextual_intent BEFORE coercion, since
    # _coerce_null_strings removes "null" strings from lists before we can detect them.
    raw_ci = data.get("contextual_intent", [])
    pre_coerce_had_null = isinstance(raw_ci, list) and any(
        isinstance(x, str) and _is_null_placeholder(x) for x in raw_ci
    )

    data = _coerce_null_strings(data)

    required = {
        "contextual_intent", "is_discriminative",
        "aspect_coverage", "disposition_note", "preference_attrs",
    }
    if not required.issubset(data):
        return None, "schema_error"

    ci = data.get("contextual_intent")
    if not isinstance(ci, list):
        return None, "schema_error"

    # Apply broader null-placeholder filter for variants not caught by _coerce_null_strings.
    ci_clean = [x for x in ci if isinstance(x, str) and not _is_null_placeholder(x)]
    null_normalized = pre_coerce_had_null or (len(ci_clean) < len(ci))
    data["contextual_intent"] = ci_clean[:2]
    data["_null_string_normalized"] = null_normalized
    if null_normalized and not ci_clean:
        data["is_discriminative"] = False

    ac = data.get("aspect_coverage") or {}
    data["_aspect_coverage_valid"] = _aspect_coverage_valid(ac)

    return data, None


def run_p1_aspect(
    client: LLMClient, item: dict, retry_max: int = 2
) -> P1CallResult:
    """Run p1_aspect for one item with two-budget retry logic.

    Budget sequence: [P1_ASPECT_DEFAULT_TOKENS, P1_RETRY_TOKENS]
    Any failure (empty / parse error / schema error / done_reason=="length")
    triggers a retry with the larger budget.
    Only stores to parsed_success_cache on clean success.
    """
    messages, truncated = build_messages(item)
    budgets = [P1_ASPECT_DEFAULT_TOKENS]
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
            messages, max_new_tokens=max_tokens, json_schema=P1_ASPECT_JSON_SCHEMA
        )
        final_tokens = max_tokens

        if not raw:
            empty_count += 1
            continue

        if done_reason == "length":
            length_count += 1

        parsed, failure = parse_response(raw)

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

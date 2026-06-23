"""P1 compact Books extractor — v0.4.3 5-field schema.

Two variants (plan.md v0.4.3 A/B):
  A — head-400-word truncation, description included if available.
  B — head-250 + tail-150 word truncation, explicit metadata fallback hierarchy,
      stronger grounding policy (metadata for disambiguation only).

JSON-schema constrained decoding via Ollama format parameter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm.client import LLMClient, P1CallResult
from src.llm.prompts.loader import load_prompt

_SYSTEM_A, _VERSION_A = load_prompt("amazon", "_default", "p1_books_a")
_SYSTEM_B, _VERSION_B = load_prompt("amazon", "_default", "p1_books_b")

P1_BOOKS_DEFAULT_TOKENS = 512
P1_BOOKS_RETRY_TOKENS = 1024

P1_BOOKS_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "contextual_intent": {"type": "string", "maxLength": 200},
        "preference_summary": {"type": "string", "maxLength": 300},
        "evidence_span": {
            "type": "array",
            "minItems": 0,
            "maxItems": 2,
            "items": {"type": "string", "maxLength": 250},
        },
        "is_discriminative": {"type": "boolean"},
        "grounding_level": {
            "type": "string",
            "enum": ["review_only", "review_plus_metadata", "metadata_dominant"],
        },
    },
    "required": [
        "contextual_intent",
        "preference_summary",
        "evidence_span",
        "is_discriminative",
        "grounding_level",
    ],
}

_USER_TEMPLATE_WITH_DESC = """\
Item title: {title}
Category: {category}
Description: {description}
Rating: {rating}/5
Review: \"\"\"{review_text}\"\"\""""

_USER_TEMPLATE_NO_DESC = """\
Item title: {title}
Category: {category}
Rating: {rating}/5
Review: \"\"\"{review_text}\"\"\""""


# ---------------------------------------------------------------------------
# Truncation helpers

def _words_head(text: str, n: int) -> str:
    """Return first n words joined by single space."""
    words = text.split()
    return " ".join(words[:n])


def _words_head_tail(text: str, head: int, tail: int) -> str:
    """Return head + '[...]' + tail if text exceeds head+tail words, else full."""
    words = text.split()
    total = head + tail
    if len(words) <= total:
        return " ".join(words)
    return " ".join(words[:head]) + " [...] " + " ".join(words[-tail:])


def _truncate_review_a(text: str) -> tuple[str, bool]:
    """Variant A: first 400 words."""
    words = text.split()
    if len(words) <= 400:
        return text, False
    return _words_head(text, 400), True


def _truncate_review_b(text: str) -> tuple[str, bool]:
    """Variant B: head-250 + tail-150 words (cap ~400)."""
    words = text.split()
    if len(words) <= 400:
        return text, False
    return _words_head_tail(text, 250, 150), True


def _category_str(category) -> str:
    if isinstance(category, list):
        return " > ".join(str(c) for c in category)
    return str(category) if category else ""


# ---------------------------------------------------------------------------
# Message builders

def build_messages_a(item: dict) -> tuple[list[dict], bool]:
    """Variant A: head-400, description if available."""
    review_text, truncated = _truncate_review_a(item.get("review_text", ""))
    description = (item.get("item_description") or "").strip()
    category = _category_str(item.get("item_category", ""))
    if description:
        user_block = _USER_TEMPLATE_WITH_DESC.format(
            title=item.get("item_title", ""),
            category=category,
            description=description[:400],
            rating=item.get("rating", ""),
            review_text=review_text,
        )
    else:
        user_block = _USER_TEMPLATE_NO_DESC.format(
            title=item.get("item_title", ""),
            category=category,
            rating=item.get("rating", ""),
            review_text=review_text,
        )
    return [
        {"role": "system", "content": _SYSTEM_A},
        {"role": "user", "content": user_block},
    ], truncated


def build_messages_b(item: dict) -> tuple[list[dict], bool]:
    """Variant B: head-250+tail-150, explicit metadata fallback."""
    review_text, truncated = _truncate_review_b(item.get("review_text", ""))
    description = (item.get("item_description") or "").strip()
    category = _category_str(item.get("item_category", ""))
    if description:
        user_block = _USER_TEMPLATE_WITH_DESC.format(
            title=item.get("item_title", ""),
            category=category,
            description=description[:400],
            rating=item.get("rating", ""),
            review_text=review_text,
        )
    else:
        user_block = _USER_TEMPLATE_NO_DESC.format(
            title=item.get("item_title", ""),
            category=category,
            rating=item.get("rating", ""),
            review_text=review_text,
        )
    return [
        {"role": "system", "content": _SYSTEM_B},
        {"role": "user", "content": user_block},
    ], truncated


def build_messages(item: dict, variant: str = "A") -> tuple[list[dict], bool]:
    if variant == "B":
        return build_messages_b(item)
    return build_messages_a(item)


# ---------------------------------------------------------------------------
# Response parsing

def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    return m.group(1) if m else text


def parse_response(raw: str) -> tuple[dict | None, str | None]:
    """Return (parsed_dict, failure_type).

    failure_type: None on success | "json_parse_error" | "schema_error"
    """
    if not raw:
        return None, "schema_error"
    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        return None, "json_parse_error"

    required = {"contextual_intent", "preference_summary", "evidence_span",
                "is_discriminative", "grounding_level"}
    if not required.issubset(data):
        return None, "schema_error"

    # Empty contextual_intent is valid only when is_discriminative=false
    # (model correctly signals nothing to extract from a generic review)
    ci = (data.get("contextual_intent") or "").strip()
    if not ci and data.get("is_discriminative") is not False:
        return None, "schema_error"

    gl = data.get("grounding_level", "")
    if gl not in ("review_only", "review_plus_metadata", "metadata_dominant"):
        return None, "schema_error"

    if not isinstance(data.get("evidence_span"), list):
        data["evidence_span"] = []

    if not isinstance(data.get("is_discriminative"), bool):
        return None, "schema_error"

    return data, None


# ---------------------------------------------------------------------------
# Main call function

def run_p1_books(
    client: LLMClient,
    item: dict,
    variant: str = "A",
    retry_max: int = 2,
) -> P1CallResult:
    """Run P1 compact Books for one item with retry logic.

    Returns a P1CallResult. The `parsed` field, when not None, contains the
    5-field compact schema dict.
    """
    # Temporarily patch the client's prompt_version so the cache key is variant-specific.
    original_version = client.config.prompt_version
    client.config.prompt_version = f"p1_books_{variant.lower()}_{_VERSION_A if variant == 'A' else _VERSION_B}"

    try:
        messages, truncated = build_messages(item, variant)
        budgets = [P1_BOOKS_DEFAULT_TOKENS]
        if retry_max >= 1:
            budgets.append(P1_BOOKS_RETRY_TOKENS)

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
                messages,
                max_new_tokens=max_tokens,
                json_schema=P1_BOOKS_JSON_SCHEMA,
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
            else:
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
    finally:
        client.config.prompt_version = original_version

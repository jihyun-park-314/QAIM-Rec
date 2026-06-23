"""P2 pseudo-query generation prompt (plan.md §5 P2).

Takes one review from a user's purchase history and generates a decontaminated
search query — the query the buyer likely typed BEFORE purchasing, phrased as a
general need with brand/model/exact spec tokens masked.

Masking validation: any title token (length >= 4, case-insensitive) found in
the output query is treated as identifying-information leakage → that query is
marked failed. Since temperature=0.0, retry would reproduce the same output, so
masking failures are excluded rather than retried (plan.md §5 P2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.llm.client import LLMClient
from src.llm.prompts.loader import load_prompt

SYSTEM_PROMPT, PROMPT_VERSION = load_prompt("amazon", "_default", "p2_pseudo_query")

P2_DEFAULT_TOKENS = 128
P2_RETRY_TOKENS = 256

_REVIEW_MAX_CHARS = 2000

P2_QUERY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 200},
                {"type": "null"},
            ]
        }
    },
    "required": ["query"],
}

_USER_TEMPLATE = """\
Item title: {title}
Category: {category}
Review: \"\"\"{review_text}\"\"\""""

_STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "for", "with", "in", "on", "at", "to",
    "of", "by", "from", "as", "is", "it", "this", "that", "my", "i", "me",
    "was", "are", "be", "been", "have", "has", "had", "will", "would", "can",
    "could", "should", "very", "also", "but", "not", "so", "if", "all",
])


@dataclass
class P2CallResult:
    query: str | None          # None if LLM returned null or failed
    masking_passed: bool       # True if query contains no title tokens
    latency_s: float
    cache_hit: bool
    done_reason: str
    retry_count: int
    failed_reason: str | None  # None on success; "llm_fail"|"parse_fail"|"mask_fail"


def _truncate_review(text: str) -> str:
    if len(text) <= _REVIEW_MAX_CHARS:
        return text
    return text[:1600] + text[-400:]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]


def _check_masking(query: str, title: str, min_token_len: int = 4) -> bool:
    """Return True (pass) if no significant title tokens appear in query.

    Compares lowercased alphanumeric tokens of length >= min_token_len,
    excluding common stop-words that carry no identifying information.
    """
    title_tokens = {
        t for t in _tokenize(title)
        if len(t) >= min_token_len and t not in _STOP_WORDS
    }
    if not title_tokens:
        return True
    query_text = query.lower()
    return not any(tok in query_text for tok in title_tokens)


def build_messages(item: dict) -> list[dict]:
    review_text = _truncate_review(item.get("review_text", ""))
    user_block = _USER_TEMPLATE.format(
        title=item.get("title", ""),
        category=item.get("category", "Books"),
        review_text=review_text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


def parse_response(raw: str) -> tuple[str | None, str | None]:
    """Return (query_str_or_None, failure_type).

    failure_type: None on success, "parse_fail" on JSON error, "null_query" if
    LLM explicitly returned null (not a failure, just no useful query found).
    """
    if not raw:
        return None, "parse_fail"
    raw = raw.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "parse_fail"
    if not isinstance(data, dict) or "query" not in data:
        return None, "parse_fail"
    q = data["query"]
    if q is None or (isinstance(q, str) and q.strip().lower() in ("null", "none", "")):
        return None, "null_query"
    if not isinstance(q, str):
        return None, "parse_fail"
    return q.strip(), None


def run_p2_query(client: LLMClient, item: dict, retry_max: int = 1) -> P2CallResult:
    """Generate a decontaminated pseudo-query for one review item.

    Masking failure is NOT retried (temperature=0 → same output) — the result
    is kept but flagged as masking_passed=False so the caller can track
    masking_pass_rate.
    """
    messages = build_messages(item)
    budgets = [P2_DEFAULT_TOKENS]
    if retry_max >= 1:
        budgets.append(P2_RETRY_TOKENS)

    latency_s = 0.0
    cache_hit = False
    done_reason = "unknown"
    retry_count = 0

    for attempt, max_tokens in enumerate(budgets):
        raw, done_reason, latency_s, cache_hit = client.generate(
            messages, max_new_tokens=max_tokens, json_schema=P2_QUERY_JSON_SCHEMA
        )
        retry_count = attempt

        if not raw:
            continue

        query, fail = parse_response(raw)
        if fail == "parse_fail":
            continue
        if fail == "null_query" or query is None:
            # LLM found no usable pre-purchase need in this review
            return P2CallResult(
                query=None,
                masking_passed=False,
                latency_s=latency_s,
                cache_hit=cache_hit,
                done_reason=done_reason,
                retry_count=retry_count,
                failed_reason="null_query",
            )

        # Successful parse — run masking check
        passed = _check_masking(query, item.get("title", ""))
        client.store_parsed(messages, json.dumps({"query": query}, ensure_ascii=False))
        return P2CallResult(
            query=query,
            masking_passed=passed,
            latency_s=latency_s,
            cache_hit=cache_hit,
            done_reason=done_reason,
            retry_count=retry_count,
            failed_reason=None if passed else "mask_fail",
        )

    # All attempts failed to parse
    return P2CallResult(
        query=None,
        masking_passed=False,
        latency_s=latency_s,
        cache_hit=cache_hit,
        done_reason=done_reason,
        retry_count=retry_count,
        failed_reason="llm_fail",
    )

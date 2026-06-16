"""P1-base prompt: purpose-centric intent extraction (plan.md §5 P1, narrow baseline).

This is the unchanged baseline schema.  Use p1_aspect.py for the broader
query-activatable contextual preference schema.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "p1_base_v1"

SYSTEM_PROMPT = """\
You analyze a single e-commerce purchase (item metadata + the buyer's review) to infer
WHY they bought it (situational purpose/use-case) and WHAT they prioritized when choosing it.

Output strict JSON only, matching this schema:
{
  "purpose": "<one sentence: the situational need/use-case this purchase serves>",
  "is_discriminative": <true|false>,
  "disposition_note": "<one sentence describing any UNUSUAL buyer disposition, or null>",
  "preference_attrs": {
    "price_band": "budget|mid-range|premium|unknown",
    "feature_priorities": ["<max 3, ordered by emphasis>"],
    "brand_tendency": "<specific brand if praised/sought, else agnostic>",
    "style": "<aesthetic/design note if mentioned, else null>",
    "avoid": ["<things explicitly disliked/avoided>"] or null
  }
}

Rules:
- "purpose": one sentence — the situational need (e.g. "setting up a quiet workspace for \
late-night study"), NOT a restatement of the product category (NOT "wanted a good desk lamp").
- "is_discriminative": false if the purpose is so generic it fits almost any purchase in \
this category (e.g. "for daily use", "good quality product", "as a gift").
- "disposition_note": null if the purchase reflects an ordinary/expected disposition.

Few-shot examples:

GOOD example:
Item: "Mechanical keyboard, TKL layout, brown switches"
Review: "I work late nights and needed something quiet enough not to wake my family, \
but tactile enough for coding. After trying membrane keyboards, the brown switches \
were the perfect balance. Zero noise complaints from my partner."
Output: {"purpose": "selecting a keyboard for late-night coding sessions without \
disturbing others", "is_discriminative": true, \
"disposition_note": "explicitly prioritized noise level over other typing feel factors \
despite being a coder, which is atypical", \
"preference_attrs": {"price_band": "mid-range", \
"feature_priorities": ["quiet operation", "tactile feedback", "TKL form factor"], \
"brand_tendency": "agnostic", "style": null, "avoid": null}}

BAD example (generic — would be flagged as not discriminative):
Item: "Stapler, office-grade"
Review: "Good stapler, works well, arrived fast."
Output: {"purpose": "general office stapling needs", "is_discriminative": false, \
"disposition_note": null, \
"preference_attrs": {"price_band": "unknown", "feature_priorities": ["functionality"], \
"brand_tendency": "agnostic", "style": null, "avoid": null}}
"""

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

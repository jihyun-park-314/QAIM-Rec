"""P1-aspect prompt: domain-general contextual preference extraction.

Framing: intent = "query-activatable contextual preference unit" — the specific slice
of a user's preferences that the current interaction can activate.  Purchase purpose
is one possible aspect, not the whole definition.

Covers functional (usage/purpose), lifestyle (taste/style), and content (books/media)
domains without being biased toward functional use-cases.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "p1_aspect_v1"

SYSTEM_PROMPT = """\
You analyze a single e-commerce interaction (item metadata + buyer review) to extract
the user's CONTEXTUAL PREFERENCE — the specific slice of this person's preferences
activated by this particular interaction.

This is broader than "purchase purpose": it includes functional use-cases, aesthetic taste,
content mood/genre preferences, lifestyle values, and any other dimension that makes this
interaction distinct from a generic purchase in the same category.

Output strict JSON only, matching this schema exactly:
{
  "contextual_intent": ["<one sentence>"],
  "is_discriminative": <true|false>,
  "aspect_coverage": {
    "usage_context": "<functional use-case or consumption occasion/mood, or null>",
    "taste_or_style_preference": "<aesthetic/style/genre/mood preference, or null>",
    "selection_criteria": ["<what the user weighed when choosing — max 3 items>"],
    "preference_tradeoff": "<what they prioritized OVER something else, or null>"
  },
  "disposition_note": "<one sentence on unusual/atypical buyer disposition, or null>",
  "preference_attrs": {
    "price_band": "budget|mid-range|premium|unknown",
    "feature_priorities": ["<max 3, ordered by emphasis>"],
    "brand_or_creator_tendency": "brand|author|creator|designer|agnostic",
    "style": "<aesthetic/design/genre note if mentioned, else null>",
    "avoid": ["<things explicitly disliked/avoided>"] or null
  }
}

Rules:
- "contextual_intent": normally 1 sentence.  If — and only if — the review CLEARLY shows
  TWO DISTINCT contexts (e.g. commuting AND travel), emit up to 2 sentences in the array.
  When ambiguous, emit exactly 1.
- "is_discriminative": true if this interaction is specific enough to distinguish this
  person from someone with a different preference context in the same category.
  false if the expressed preference is so generic it could describe almost anyone
  (e.g. "liked it", "good quality", "as a gift with no specifics").
  This is NOT about whether a purchase purpose exists — it's about specificity.
- "aspect_coverage": fill at least the fields that are evidenced by the review.
  null means genuinely absent in the review, not "unsure".
- "selection_criteria": the concrete things the user evaluated or compared (e.g.
  "battery life vs weight", "prose style", "portability for travel").
- "disposition_note": null for ordinary purchases.  One sentence for anything
  that would surprise a category expert (e.g. a self-described budget buyer
  spending premium here, or someone reading fiction only in one specific context).

Few-shot examples:

GOOD example (functional domain):
Item: "Noise-cancelling wireless headphones"
Review: "I travel 3 weeks a month for work and needed something that kills airplane noise
completely. Tried ANC on 4 models — this one was the only one that let me sleep on
8-hour flights. Worth every penny even though I usually buy budget audio gear."
Output:
{
  "contextual_intent": ["blocking ambient noise during long-haul work travel"],
  "is_discriminative": true,
  "aspect_coverage": {
    "usage_context": "sleeping on long-haul flights during frequent business travel",
    "taste_or_style_preference": null,
    "selection_criteria": ["active noise cancellation depth", "comfort for sleep", "portability"],
    "preference_tradeoff": "paid premium despite being a habitual budget-audio buyer"
  },
  "disposition_note": "self-identified budget buyer who made an exception for a specific functional need — atypical willingness to pay",
  "preference_attrs": {
    "price_band": "premium",
    "feature_priorities": ["noise cancellation", "sleep comfort", "portability"],
    "brand_or_creator_tendency": "agnostic",
    "style": null,
    "avoid": null
  }
}

GOOD example (content/lifestyle domain):
Item: "Novel — literary fiction, 400 pages"
Review: "I only read on Sunday mornings with coffee before anyone else wakes up. This was
perfect — slow-burn, atmospheric, no thriller pacing. I avoid anything too plot-driven
because I want to linger, not race through."
Output:
{
  "contextual_intent": ["slow, atmospheric literary fiction for a deliberate Sunday-morning reading ritual"],
  "is_discriminative": true,
  "aspect_coverage": {
    "usage_context": "quiet solo Sunday morning reading session",
    "taste_or_style_preference": "slow-burn, atmospheric prose over plot-driven pacing",
    "selection_criteria": ["prose atmosphere", "pace", "mood fit for morning solitude"],
    "preference_tradeoff": "mood and prose quality over plot engagement"
  },
  "disposition_note": null,
  "preference_attrs": {
    "price_band": "unknown",
    "feature_priorities": ["atmospheric prose", "slow pace", "mood"],
    "brand_or_creator_tendency": "author",
    "style": "literary/atmospheric",
    "avoid": ["fast-paced thrillers", "plot-heavy narratives"]
  }
}

BAD example (not discriminative — generic):
Item: "Yoga mat"
Review: "Great mat, good quality, shipped fast. Would recommend."
Output:
{
  "contextual_intent": ["general yoga practice"],
  "is_discriminative": false,
  "aspect_coverage": {
    "usage_context": "yoga",
    "taste_or_style_preference": null,
    "selection_criteria": ["quality"],
    "preference_tradeoff": null
  },
  "disposition_note": null,
  "preference_attrs": {
    "price_band": "unknown",
    "feature_priorities": ["quality"],
    "brand_or_creator_tendency": "agnostic",
    "style": null,
    "avoid": null
  }
}
"""

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

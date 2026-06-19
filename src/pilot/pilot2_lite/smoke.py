"""Task 2-B: Multi-intent smoke — LLM, 10 users × 3 domains.

Uses p1_aspect (v4 = current stable prompt) via existing LLMClient.
p1_base is NOT used.

Reports estimated call count and time before running.
Max 240 LLM calls total (10 users × 3 domains × 8 interactions).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient, load_llm_config
from src.llm.prompts import p1_aspect


_LLM_CONFIG_PATH = "configs/llm/p1_aspect.yaml"
_ESTIMATED_SECONDS_PER_CALL = 8.0  # conservative estimate (cache-miss)


def estimate_calls(
    user_data_per_domain: dict[str, list[dict]],
) -> dict:
    total_calls = 0
    per_domain = {}
    for domain, users in user_data_per_domain.items():
        n_calls = sum(u["selected_interaction_count"] for u in users)
        total_calls += n_calls
        per_domain[domain] = n_calls
    estimated_seconds = total_calls * _ESTIMATED_SECONDS_PER_CALL
    return {
        "total_estimated_calls": total_calls,
        "per_domain": per_domain,
        "estimated_seconds": round(estimated_seconds, 0),
        "estimated_minutes": round(estimated_seconds / 60, 1),
    }


def run_smoke_for_domain(
    domain: str,
    users: list[dict],
    domain_type: str = "lifestyle",
) -> list[dict[str, Any]]:
    """Run p1_aspect for all interactions of the given users.

    Returns per-user result dicts with LLM outputs attached.
    """
    llm_cfg = load_llm_config(_LLM_CONFIG_PATH)
    llm_cfg.retry_max = 2
    llm_cfg.prompt_version = p1_aspect.PROMPT_VERSION
    client = LLMClient(llm_cfg)

    results = []
    for user in users:
        uid = user["user_id"]
        interactions = user["eligible_train_interactions"]
        print(f"  [smoke] {domain} user={uid[:12]}... ({len(interactions)} interactions)")

        per_interaction = []
        for it in interactions:
            item = {
                "title": it["title"],
                "category": it["category"],
                "brand": it["brand"],
                "price": it["price"],
                "rating": it["rating"],
                "review_text": it["review_text"],
                "parent_asin": it["parent_asin"],
                "user_id": uid,
            }
            t0 = time.time()
            res = p1_aspect.run_p1_aspect(client, item, retry_max=2)
            latency = round(time.time() - t0, 3)

            per_interaction.append({
                "parent_asin": it["parent_asin"],
                "title": it["title"],
                "timestamp": it["timestamp"],
                "rating": it["rating"],
                "parse_success": res.parsed is not None,
                "cache_hit": res.cache_hit,
                "latency_s": latency,
                "retry_count": res.retry_count,
                "llm_output": res.parsed,
            })

        results.append({
            "user_id": uid,
            "domain": domain,
            "domain_type": domain_type,
            "total_eligible_train_count": user["total_eligible_train_count"],
            "selected_interaction_count": user["selected_interaction_count"],
            "interactions": per_interaction,
        })

    client.close()
    return results

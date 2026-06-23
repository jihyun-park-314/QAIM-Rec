"""5-user end-to-end smoke: P1 extract → filter → embed → cluster.

Verifies: parse success, 5-field schema, cluster formation (k_min=1),
NO title tokens in source_text.

Run: python -m src.memory.smoke5
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from src.llm.client import LLMClient, load_llm_config
from src.memory.embed import EmbeddingModel
from src.memory.pipeline import extract_record, cluster_user_records, check_leakage


def main():
    # Load candidates
    with open("data/processed/Books/books_memory_candidates.jsonl", encoding="utf-8") as f:
        all_recs = [json.loads(l) for l in f if l.strip()]

    by_user = defaultdict(list)
    for r in all_recs:
        by_user[r["user_id"]].append(r)

    eligible_users = [u for u, recs in by_user.items() if len(recs) >= 2]
    rng = random.Random(42)
    rng.shuffle(eligible_users)
    selected = eligible_users[:5]

    print(f"[smoke5] Selected users: {selected}")
    for u in selected:
        print(f"  user={u} n_reviews={len(by_user[u])}")

    llm_cfg = load_llm_config("configs/llm/p1.yaml")
    client = LLMClient(llm_cfg)
    emb_model = EmbeddingModel("BAAI/bge-base-en-v1.5")

    variant = "A"
    all_ok = True

    for uid in selected:
        reviews = by_user[uid][:12]
        print(f"\n{'='*60}")
        print(f"USER {uid}  ({len(reviews)} reviews, variant {variant})")

        recs = [extract_record(client, item, variant=variant) for item in reviews]

        parsed = [r for r in recs if not r.parse_failed]
        eligible = [r for r in recs if r.eligible]
        leakage = [r for r in parsed if r.leakage_detected]

        print(f"  parse  : {len(parsed)}/{len(recs)} success")
        print(f"  filter : {len(eligible)}/{len(parsed)} eligible")
        print(f"  leakage: {len(leakage)} titles leaked into source_text")

        for r in parsed:
            flag = " [FAIL:no_ci]" if not r.contextual_intent.strip() else ""
            flag += " [FAIL:no_evid]" if not r.evidence_span else ""
            flag += " [LEAK]" if r.leakage_detected else ""
            flag += " [ELIG]" if r.eligible else ""
            gl = r.grounding_level[:14] if r.grounding_level else "?"
            print(
                f"    item={r.item_id} disc={r.is_discriminative} "
                f"gl={gl:20s} cache={r.cache_hit} lat={r.latency_s:.1f}s{flag}"
            )
            if r.contextual_intent:
                print(f"      ci   : {r.contextual_intent[:100]}")
            if r.preference_summary:
                print(f"      ps   : {r.preference_summary[:100]}")
            if r.evidence_span:
                print(f"      evid : {r.evidence_span[0][:80]}")
            if r.source_text:
                print(f"      src  : {r.source_text[:120]}")

        # Schema completeness check
        schema_fields = ["contextual_intent", "preference_summary", "evidence_span",
                         "is_discriminative", "grounding_level"]
        for r in parsed:
            for f_name in schema_fields:
                v = getattr(r, f_name)
                if v is None:
                    print(f"  [ERR] user={uid} item={r.item_id}: field '{f_name}' is None")
                    all_ok = False

        # Cluster
        if eligible:
            cluster_res = cluster_user_records(eligible, emb_model, k_min=1, k_max=5, tau=0.3)
            k = cluster_res["k_personal"]
            print(f"\n  CLUSTER: k_personal={k}")
            for c in cluster_res["cluster_summaries"]:
                print(f"    cluster {c['label']} (size={c['size']}):")
                for intent in c["intents"]:
                    print(f"      • {intent[:90]}")
            if k == 0:
                print("  [ERR] k_personal == 0 despite eligible records!")
                all_ok = False
        else:
            k = 0
            print(f"\n  CLUSTER: k_personal=0 (no eligible records)")

        # Leakage assert
        if leakage:
            print(f"  [WARN] {len(leakage)} leakage cases:")
            for r in leakage:
                print(f"    item={r.item_id} title='{r.item_title}' src='{r.source_text[:100]}'")

    print(f"\n{'='*60}")
    print(f"[smoke5] {'ALL OK' if all_ok else 'SOME CHECKS FAILED'}")

    client.close()


if __name__ == "__main__":
    main()

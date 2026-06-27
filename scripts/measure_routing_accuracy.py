"""STEP 3 Measurement 1 — Pre-training provenance routing accuracy.

For each query variant in ablation_p2.json:
  query → embed → cosine-sim against user's memory unit vectors → top-1 hit?
  Compare top-1 memory_id vs provenance GT (positive_memory_id).

Reports per variant: overall Hits@1, stratified by K=1 / K>=2.
No LLM calls. Embedding model: BAAI/bge-base-en-v1.5 (matches memory bank).

Usage:
  python3 scripts/measure_routing_accuracy.py \\
    --ablation  data/processed/Books/ablation_p2.json \\
    --bank_dir  data/processed/Books/memory_bank \\
    --p1_extractions data/processed/Books/p1_extractions.jsonl \\
    --selected_review_ids data/p1_shards_gpu01/selected_review_ids.json \\
                          data/p1_shards_gpu23/selected_review_ids.json \\
    --output    data/processed/Books/routing_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_pseudo_queries import (
    load_memory_bank,
    load_p1_index,
    load_selected_ids,
)


# ---------------------------------------------------------------------------
# Memory bank loader — returns per-user {memory_id: vector} + k_personal

def load_bank_vectors(bank_dir: str) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, int]]:
    """Returns:
      unit_vecs  {uid: {memory_id: np.ndarray [768]}}
      k_map      {uid: k_personal}
    """
    unit_vecs: dict[int, dict[str, np.ndarray]] = {}
    k_map: dict[int, int] = {}

    for p in sorted(Path(bank_dir).glob("[0-9]*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        uid = int(data["user_id"])
        k = data.get("k_personal", 0)
        if k < 1:
            continue
        k_map[uid] = k
        vecs: dict[str, np.ndarray] = {}
        for unit in data.get("units", []):
            mid = unit["memory_id"]
            emb = unit.get("embedding", {})
            vec = emb.get("vector")
            if vec:
                vecs[mid] = np.array(vec, dtype=np.float32)
        unit_vecs[uid] = vecs

    return unit_vecs, k_map


# ---------------------------------------------------------------------------
# Embedding model (uses transformers directly — sentence_transformers not required)

def load_embed_model(model_id: str = "BAAI/bge-base-en-v1.5"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    # Pick the GPU with most free memory; fall back to CPU
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        device = f"cuda:{free.index(max(free))}"
    else:
        device = "cpu"
    model = model.to(device).eval()
    print(f"  embedding device: {device}", flush=True)
    return (model, tokenizer, device)


def embed_queries(model_bundle, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Returns L2-normalized embeddings [N, 768] via mean pooling."""
    import torch
    model, tokenizer, device = model_bundle
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # mean pooling
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0)


# ---------------------------------------------------------------------------
# Routing accuracy for one variant

def evaluate_variant(
    records: list[dict],
    unit_vecs: dict[int, dict[str, np.ndarray]],
    k_map: dict[int, int],
    provenance_index: dict[int, dict[int, str]],
    embed_model,
) -> dict:
    # Collect non-null queries that have provenance GT
    eval_rows = []
    for r in records:
        if not r.get("query"):
            continue
        uid = r["uid"]
        iid = r["iid"]
        prov = provenance_index.get(uid, {})
        pos_mid = prov.get(iid)
        if pos_mid is None:
            continue
        if uid not in unit_vecs:
            continue
        eval_rows.append((uid, iid, r["query"], pos_mid))

    if not eval_rows:
        return {"n": 0, "hits1_all": None, "hits1_k1": None, "hits1_kge2": None}

    # Batch-embed all queries
    queries = [row[2] for row in eval_rows]
    q_vecs = embed_queries(embed_model, queries)  # [N, 768]

    hits_all, hits_k1, hits_kge2 = [], [], []

    for i, (uid, iid, query, pos_mid) in enumerate(eval_rows):
        user_mids = list(unit_vecs[uid].keys())
        mem_matrix = np.stack([unit_vecs[uid][m] for m in user_mids])  # [K, 768]
        q_vec = q_vecs[i]  # [768], already L2-normalized

        # Cosine sim (both already L2-normalized)
        sims = mem_matrix @ q_vec  # [K]
        top1_mid = user_mids[int(np.argmax(sims))]
        hit = int(top1_mid == pos_mid)

        k = k_map.get(uid, 1)
        hits_all.append(hit)
        if k == 1:
            hits_k1.append(hit)
        else:
            hits_kge2.append(hit)

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "n": len(hits_all),
        "n_k1": len(hits_k1),
        "n_kge2": len(hits_kge2),
        "hits1_all": _mean(hits_all),
        "hits1_k1": _mean(hits_k1),
        "hits1_kge2": _mean(hits_kge2),
    }


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--p1_extractions", required=True)
    ap.add_argument("--selected_review_ids", nargs="+", required=True)
    ap.add_argument("--output", default="data/processed/Books/routing_accuracy.json")
    ap.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    print("Loading memory bank vectors ...", flush=True)
    unit_vecs, k_map = load_bank_vectors(args.bank_dir)
    provenance_index, _ = load_memory_bank(args.bank_dir)
    print(f"  users with K>=1: {len(unit_vecs)}", flush=True)

    print("Loading embedding model ...", flush=True)
    embed_model = load_embed_model(args.embed_model)

    print("Loading ablation results ...", flush=True)
    with open(args.ablation, encoding="utf-8") as f:
        ablation = json.load(f)

    print(f"\nEvaluating {len(ablation)} variants ...", flush=True)
    results = []
    for variant in sorted(ablation, key=lambda x: x["code"]):
        code = variant["code"]
        metrics = evaluate_variant(
            variant["records"], unit_vecs, k_map, provenance_index, embed_model
        )
        metrics["code"] = code
        metrics["null_rate"] = variant["null_rate"]
        metrics["title_leak_rate"] = variant["title_leak_rate"]
        metrics["author_leak_rate"] = variant["author_leak_rate"]
        results.append(metrics)
        print(
            f"  {code}  n={metrics['n']:3d}  "
            f"Hits@1={metrics['hits1_all']:.3f}  "
            f"K=1:{metrics['hits1_k1']}  "
            f"K>=2:{metrics['hits1_kge2']}",
            flush=True,
        )

    # Summary table
    print(f"\n{'='*65}")
    print("  Routing accuracy (Hits@1) — pre-training embedding baseline")
    print(f"  Columns: L A G T  |  Hits@1(all)  K=1  K>=2  |  null%  author_lk%")
    print(f"{'='*65}")
    for r in results:
        h_all = f"{r['hits1_all']:.3f}" if r["hits1_all"] is not None else "  N/A"
        h_k1  = f"{r['hits1_k1']:.3f}"  if r["hits1_k1"]  is not None else "  N/A"
        h_k2  = f"{r['hits1_kge2']:.3f}" if r["hits1_kge2"] is not None else "  N/A"
        print(
            f"  {r['code']}   {h_all}        {h_k1}  {h_k2}  "
            f"|  {r['null_rate']*100:4.1f}%  {r['author_leak_rate']*100:4.1f}%"
        )
    print(f"{'='*65}")
    print("  Note: K=1 users always Hits@1=1.0 (only one memory unit).")
    print("  Meaningful signal comes from K>=2 rows.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {args.output}", flush=True)


if __name__ == "__main__":
    main()

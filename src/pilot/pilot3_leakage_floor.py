"""Pilot 3 — Leakage floor measurement (plan.md §4 Stage P3, F8 reference baseline).

Pipeline: text-encode(query) → cosine vs all item embeddings → rank test target
→ Recall@{5,10,20} = leakage floor.  No SASRec, no steering.

Interpretation: the floor should be LOW.  A high floor means query-alone already
retrieves the test item — steering adds no distinguishable lift.  F8 reports
(correct_condition_Recall − floor) as the credible steering contribution.

Usage — CPU smoke (20-user bank, mock queries):
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        src/pilot/pilot3_leakage_floor.py --smoke \\
        --bank_jsonl data/memory_full_test/memory_b_u20_seed42.jsonl \\
        --splits data/processed/Books/splits.json

Usage — full run (after P2 generation):
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        src/pilot/pilot3_leakage_floor.py \\
        --pseudo_queries data/processed/Books/pseudo_queries.jsonl \\
        --splits data/processed/Books/splits.json \\
        --id_maps data/processed/Books/id_maps.json \\
        --item_meta data/raw/Books/meta.jsonl \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_KS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Text encoder (inference-only, wraps EmbeddingModel for consistency)

class _TextEncoder:
    """CPU/GPU text encoder using bge-base.  Outputs numpy float32 [N, d]."""

    def __init__(self, device: str = "cpu") -> None:
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(_BGE_MODEL_ID, local_files_only=True)
        self._model = AutoModel.from_pretrained(_BGE_MODEL_ID, local_files_only=True)
        self._model.eval().to(device)
        self.device = device

    def encode(self, texts: list[str], is_query: bool = False,
               batch_size: int = 64) -> np.ndarray:
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs = F.normalize(vecs, p=2, dim=-1)
            all_vecs.append(vecs.cpu().numpy())
        return np.concatenate(all_vecs, axis=0)  # [N, d]


# ---------------------------------------------------------------------------
# Data loaders

def _load_splits(splits_path: str) -> dict:
    """splits.json → {user_id(int): {train:[...], val:int, test:int}}."""
    with open(splits_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(uid): v for uid, v in raw["users"].items()}, raw["meta"]


def _load_id_maps(id_maps_path: str) -> dict:
    """id_maps.json → id2item {int → asin_str}."""
    with open(id_maps_path, encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d["id2item"].items()}


def _stream_item_titles(meta_jsonl: str, needed_asins: set) -> dict:
    """Stream meta.jsonl → {asin: title} for needed ASINs."""
    asin2title: dict[str, str] = {}
    with open(meta_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin", "")
            if asin in needed_asins:
                title = rec.get("title", "")
                if title:
                    asin2title[asin] = title
            if len(asin2title) >= len(needed_asins):
                break
    return asin2title


# ---------------------------------------------------------------------------
# Core leakage floor computation

def compute_leakage_floor(
    queries: list[str],        # [M] — one per user
    user_ids: list[int],       # [M]
    item_titles: list[str],    # [N] — full catalog titles
    item_ids: list[int],       # [N] — catalog item_ids (1-indexed)
    splits: dict,              # {user_id: {train:[...], val:int, test:int}}
    encoder: _TextEncoder,
    ks: list[int] = None,
) -> dict:
    """Return {Recall@k, NDCG@k, MRR@k for k in ks, n_eval, floor_judgment}."""
    ks = ks or _KS
    print(f"[leakage] encoding {len(queries)} queries ...", flush=True)
    q_emb = encoder.encode(queries, is_query=True)   # [M, d]
    print(f"[leakage] encoding {len(item_titles)} items ...", flush=True)
    i_emb = encoder.encode(item_titles, is_query=False)  # [N, d]

    iid2row = {iid: idx for idx, iid in enumerate(item_ids)}
    scores_mat = q_emb @ i_emb.T   # [M, N]

    recall = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = {k: 0.0 for k in ks}
    n_eval = 0

    for qi, (uid, qscores) in enumerate(zip(user_ids, scores_mat)):
        udata = splits.get(uid)
        if udata is None:
            continue
        test_iid = udata.get("test")
        if not test_iid or test_iid not in iid2row:
            continue

        # Mask seen items
        seen_iids = set(udata.get("train", []))
        val_iid = udata.get("val")
        if val_iid:
            seen_iids.add(val_iid)
        seen_rows = np.array([iid2row[i] for i in seen_iids if i in iid2row])
        if len(seen_rows) > 0:
            qscores = qscores.copy()
            qscores[seen_rows] = -1e9

        target_row = iid2row[test_iid]
        target_score = qscores[target_row]
        rank = int((qscores > target_score).sum())  # 0-indexed

        n_eval += 1
        for k in ks:
            if rank < k:
                recall[k] += 1
                ndcg[k] += 1.0 / np.log2(rank + 2)
                mrr[k] += 1.0 / (rank + 1)

    if n_eval == 0:
        metrics = {f"{m}@{k}": 0.0 for m in ["Recall", "NDCG", "MRR"] for k in ks}
    else:
        metrics = {}
        for k in ks:
            metrics[f"Recall@{k}"] = round(recall[k] / n_eval, 6)
            metrics[f"NDCG@{k}"] = round(ndcg[k] / n_eval, 6)
            metrics[f"MRR@{k}"] = round(mrr[k] / n_eval, 6)

    # Floor judgment: Recall@10 < 0.05 is "floor is low → steering lift credible"
    floor_low = metrics.get("Recall@10", 0) < 0.05
    metrics["n_eval"] = n_eval
    metrics["floor_judgment"] = "LOW (steering lift credible)" if floor_low else \
        "HIGH (query alone strong — interpret steering lift vs this floor carefully)"
    return metrics


# ---------------------------------------------------------------------------
# Smoke runner

def run_smoke(args: SimpleNamespace) -> None:
    """CPU smoke: mock queries from bank source_texts, small mock catalog."""
    print("[smoke] Loading bank (source_text as mock query) ...")
    bank_users: list[dict] = []
    with open(args.bank_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts = []
            for cs in rec.get("cluster_summaries", []):
                texts.extend(cs.get("source_texts", []))
            if texts:
                bank_users.append({"user_id": rec["user_id"], "source_text": texts[0]})
            if len(bank_users) >= 5:
                break

    assert len(bank_users) >= 2, "Need at least 2 users for smoke"
    print(f"[smoke] {len(bank_users)} mock users: "
          f"{[u['user_id'] for u in bank_users]}")

    print("[smoke] Loading splits.json ...")
    splits_raw, meta = _load_splits(args.splits)
    n_items = meta["n_items"]
    print(f"[smoke] {meta['n_users']} users, {n_items} items")

    # Build small mock catalog: test targets + random fillers
    user_ids = [u["user_id"] for u in bank_users]
    queries = [u["source_text"][:200] for u in bank_users]

    target_iids = set()
    for uid in user_ids:
        udata = splits_raw.get(uid)
        if udata:
            t = udata.get("test")
            if t:
                target_iids.add(t)

    # Fill catalog: test targets first, then random items up to 50
    import random
    rng = random.Random(42)
    filler_iids = rng.sample(
        [i for i in range(1, n_items + 1) if i not in target_iids],
        min(50 - len(target_iids), n_items - len(target_iids))
    )
    catalog_iids = sorted(target_iids) + filler_iids
    catalog_titles = [f"book item {iid} about fiction and literature" for iid in catalog_iids]
    print(f"[smoke] Mock catalog: {len(catalog_iids)} items "
          f"(targets={len(target_iids)}, fillers={len(filler_iids)})")

    print("[smoke] Loading bge-base encoder (CPU) ...")
    encoder = _TextEncoder(device="cpu")

    print("[smoke] Computing leakage floor ...")
    metrics = compute_leakage_floor(
        queries=queries,
        user_ids=user_ids,
        item_titles=catalog_titles,
        item_ids=catalog_iids,
        splits=splits_raw,
        encoder=encoder,
    )

    print("\n[smoke] === Leakage Floor (smoke, mock data — not real results) ===")
    for k in _KS:
        r = metrics.get(f"Recall@{k}", 0)
        n = metrics.get(f"NDCG@{k}", 0)
        m = metrics.get(f"MRR@{k}", 0)
        print(f"  @{k:2d}: Recall={r:.4f}  NDCG={n:.4f}  MRR={m:.4f}")
    print(f"  n_eval = {metrics['n_eval']}")
    print(f"  floor_judgment: {metrics['floor_judgment']}")
    print("\n[smoke] NOTE: mock catalog titles are meaningless — "
          "real run uses actual book titles from meta.jsonl")
    print("[smoke] ALL CHECKS PASSED (CPU, mock data — not real results)")


# ---------------------------------------------------------------------------
# Full run

def run_full(args: SimpleNamespace) -> None:
    """Full leakage floor: pseudo_queries.jsonl + real item titles."""
    print(f"[leakage] Loading pseudo queries from {args.pseudo_queries} ...")
    queries: list[str] = []
    user_ids: list[int] = []
    with open(args.pseudo_queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("query")
            if q and rec.get("masking_passed", True):
                queries.append(q)
                user_ids.append(int(rec["user_id"]))
    print(f"[leakage] {len(queries)} valid queries (masking_passed=True)")

    print(f"[leakage] Loading splits from {args.splits} ...")
    splits_raw, meta = _load_splits(args.splits)
    n_items = meta["n_items"]

    print(f"[leakage] Loading id_maps from {args.id_maps} ...")
    id2item = _load_id_maps(args.id_maps)

    print(f"[leakage] Streaming item titles from {args.item_meta} ...")
    needed_asins = set(id2item.values())
    asin2title = _stream_item_titles(args.item_meta, needed_asins)
    print(f"[leakage] item titles found: {len(asin2title)}/{len(needed_asins)}")

    catalog_iids = sorted(id2item.keys())
    catalog_titles = [asin2title.get(id2item[iid], f"book {iid}") for iid in catalog_iids]

    encoder = _TextEncoder(device=args.device)

    metrics = compute_leakage_floor(
        queries=queries,
        user_ids=user_ids,
        item_titles=catalog_titles,
        item_ids=catalog_iids,
        splits=splits_raw,
        encoder=encoder,
    )

    print("\n=== Leakage Floor (full run) ===")
    for k in _KS:
        r = metrics.get(f"Recall@{k}", 0)
        n = metrics.get(f"NDCG@{k}", 0)
        m = metrics.get(f"MRR@{k}", 0)
        print(f"  @{k:2d}: Recall={r:.4f}  NDCG={n:.4f}  MRR={m:.4f}")
    print(f"  n_eval = {metrics['n_eval']}")
    print(f"  floor_judgment: {metrics['floor_judgment']}")
    print()
    print("NOTE: floor 낮아야 steering 신뢰 — F8에서 (correct_Recall − floor)가 "
          "실질 steering lift이다.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[leakage] results written to {out}")


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="CPU smoke with mock queries from bank source_texts")
    parser.add_argument("--bank_jsonl", type=str,
                        default="data/memory_full_test/memory_b_u20_seed42.jsonl",
                        help="Smoke: 20-user bank JSONL")
    parser.add_argument("--splits", type=str,
                        default="data/processed/Books/splits.json")
    parser.add_argument("--pseudo_queries", type=str,
                        default="data/processed/Books/pseudo_queries.jsonl",
                        help="Full run: P2-generated queries")
    parser.add_argument("--id_maps", type=str,
                        default="data/processed/Books/id_maps.json")
    parser.add_argument("--item_meta", type=str,
                        default="data/raw/Books/meta.jsonl",
                        help="Full run: meta.jsonl for item titles")
    parser.add_argument("--output", type=str,
                        default="results/pilot/pilot3_leakage_floor.json")
    parser.add_argument("--device", type=str, default="cpu")
    cli = parser.parse_args()

    if cli.smoke:
        run_smoke(cli)
    else:
        run_full(cli)


if __name__ == "__main__":
    main()

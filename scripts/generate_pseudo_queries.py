"""Generate P2 pseudo-queries — dual mode: train (per history review) + eval (per test review).

MODE train:
  Reads books_memory_candidates.jsonl (train-history reviews only) for personal users
  (k_personal >= 1). Generates one BLAIR-style search query per review.
  memory_id is null — P3 assigns memory provenance by cosine similarity at align-pair time.
  Output used to build P3 align-pairs for contrastive training.

MODE eval:
  Reads raw reviews for each user's test item (from splits.json), generates one
  search query per user. NEVER mixed into training. Isolation guaranteed by splits.

Output schema (one JSON per line):
  {
    "user_id":      int,
    "source":       "train" | "eval",
    "memory_id":    str | null,   # always null for train (P3 assigns); null for eval
    "target_item":  int | null,   # eval: test item_id; train: null
    "query_text":   str | null,   # null if LLM returned null or failed
    "src_review_ids": [int, ...], # train: [item_id of this review]; eval: [test item_id]
    "masking_passed": bool,
    "failed_reason": str | null   # "null_query" | "llm_fail" | "mask_fail" | "no_review" | null
  }

Usage — train:
  python3 -u scripts/generate_pseudo_queries.py --mode train \\
    --bank_dir data/processed/Books/memory_bank \\
    --candidates_jsonl data/processed/Books/books_memory_candidates.jsonl \\
    --prompt_train config/prompts/amazon/Books/p2_pseudo_query_train.txt \\
    --output data/processed/Books/pseudo_queries_train.jsonl \\
    --llm_config configs/llm/p2.yaml

Usage — eval:
  python3 -u scripts/generate_pseudo_queries.py --mode eval \\
    --splits_json data/processed/Books/splits.json \\
    --id_maps_json data/processed/Books/id_maps.json \\
    --reviews_jsonl data/raw/Books/reviews.jsonl \\
    --prompt_eval config/prompts/amazon/Books/p2_pseudo_query_eval.txt \\
    --output data/processed/Books/pseudo_queries_eval.jsonl \\
    --llm_config configs/llm/p2.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from",
              "into", "have", "been", "were", "they", "their", "what",
              "book", "books", "read", "reading", "story", "novel"}


# ---------------------------------------------------------------------------
# Shared helpers

def _masking_passed(query: str, title: str) -> bool:
    """True if no significant title tokens appear in query."""
    title_tokens = set(re.findall(r"[a-zA-Z0-9]{4,}", title.lower()))
    title_tokens -= _STOPWORDS
    if not title_tokens:
        return True
    query_lower = query.lower()
    return not any(tok in query_lower for tok in title_tokens)


def _parse_query(raw: str) -> tuple[str | None, str | None]:
    """Parse LLM output. Returns (query_text, failed_reason)."""
    m = re.search(r'\{[^{}]*?"query"\s*:[^{}]*?\}', raw, re.DOTALL)
    if not m:
        # Fallback: try to extract any JSON object
        m = re.search(r'\{.*?"query".*?\}', raw, re.DOTALL)
    if not m:
        return None, "llm_fail"
    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError:
        return None, "llm_fail"
    query = parsed.get("query")
    if query is None:
        return None, "null_query"
    query = str(query).strip()
    if not query:
        return None, "null_query"
    return query, None


def _llm_call(client, prompt: str, retry_max: int = 2) -> tuple[str | None, str | None]:
    """Call LLM, parse JSON, return (query_text, failed_reason)."""
    for _ in range(retry_max + 1):
        try:
            raw, _, _, _ = client.generate([{"role": "user", "content": prompt}])
            query, reason = _parse_query(raw)
            if reason != "llm_fail":
                return query, reason
        except Exception:
            pass
    return None, "llm_fail"


# ---------------------------------------------------------------------------
# TRAIN mode

def _build_personal_uids(bank_dir: str) -> set[int]:
    """Return set of user_ids whose fallback_type == 'personal' (k_personal >= 1)."""
    personal = set()
    for p in Path(bank_dir).glob("[0-9]*.json"):
        rec = json.load(open(p, encoding="utf-8"))
        if rec.get("fallback_type") == "personal":
            personal.add(int(rec["user_id"]))
    return personal


def _iter_train(candidates_jsonl: str, bank_dir: str):
    """Yield (user_id, item_id, review_text, item_title) for personal users' train reviews."""
    print("[p2_train] Building personal user set from bank ...", flush=True)
    personal_uids = _build_personal_uids(bank_dir)
    print(f"[p2_train] Personal users: {len(personal_uids)}", flush=True)

    with open(candidates_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uid = int(r["user_id"])
            if uid not in personal_uids:
                continue
            review_text = r.get("review_text", "").strip()
            if not review_text:
                continue
            yield uid, int(r["item_id"]), review_text, r.get("item_title", "")


def run_train(client, candidates_jsonl: str, bank_dir: str, prompt_tmpl: str,
              out_path: Path, max_records: int, dry_run: int) -> None:
    n_ok = n_null = n_fail = n_mask = 0
    limit = dry_run if dry_run > 0 else max_records

    with open(out_path, "w", encoding="utf-8") as fout:
        for user_id, item_id, review_text, item_title in _iter_train(candidates_jsonl, bank_dir):
            if n_ok + n_null + n_fail + n_mask >= limit:
                break

            prompt = (prompt_tmpl
                      .replace("{title}", item_title)
                      .replace("{review_text}", review_text[:1000]))

            query, reason = _llm_call(client, prompt)

            masking_passed = True
            if query and item_title and reason is None:
                masking_passed = _masking_passed(query, item_title)
                if not masking_passed:
                    reason = "mask_fail"
                    query2, reason2 = _llm_call(client, prompt, retry_max=1)
                    if query2 and _masking_passed(query2, item_title):
                        query, reason, masking_passed = query2, None, True
                    else:
                        n_mask += 1

            if reason == "null_query":
                n_null += 1
            elif reason == "llm_fail":
                n_fail += 1
            elif reason == "mask_fail":
                pass
            else:
                n_ok += 1

            rec = {
                "user_id": int(user_id),
                "source": "train",
                "memory_id": None,
                "target_item": None,
                "query_text": query,
                "src_review_ids": [item_id],
                "masking_passed": masking_passed,
                "failed_reason": reason,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total = n_ok + n_null + n_fail + n_mask
            if total % 500 == 0 and total > 0:
                print(f"  [train {total}] ok={n_ok} null={n_null} fail={n_fail} mask={n_mask}",
                      flush=True)

    total = n_ok + n_null + n_fail + n_mask
    print(f"[p2_train] done: total={total} ok={n_ok} null={n_null} fail={n_fail} "
          f"mask={n_mask}  pass_rate={n_ok/max(total,1):.3f}")


# ---------------------------------------------------------------------------
# EVAL mode

def _build_review_index(reviews_jsonl: str, user2id: dict, item2id: dict) -> dict:
    """Build {internal_user_id: {internal_item_id: (review_text, item_title)}} index.

    Streams the raw reviews file once. Only keeps entries where both user and item
    are in our id maps (i.e., in the processed dataset).
    """
    print("[p2_eval] Building review index from raw reviews ...", flush=True)
    idx: dict[int, dict[int, tuple[str, str]]] = {}
    n = 0
    with open(reviews_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            orig_uid = r.get("user_id", "")
            orig_iid = r.get("parent_asin", "")
            if orig_uid not in user2id or orig_iid not in item2id:
                continue
            uid = user2id[orig_uid]
            iid = item2id[orig_iid]
            text = r.get("text", "").strip()
            title = r.get("title", "").strip()
            if not text:
                continue
            if uid not in idx:
                idx[uid] = {}
            idx[uid][iid] = (text, title)
            n += 1
    print(f"[p2_eval] Index built: {n} reviews for {len(idx)} users", flush=True)
    return idx


def run_eval(client, splits_json: str, id_maps_json: str, reviews_jsonl: str,
             prompt_tmpl: str, out_path: Path, max_records: int, dry_run: int) -> None:
    with open(splits_json) as f:
        splits_data = json.load(f)
    splits = splits_data.get("users", splits_data) if "users" in splits_data else splits_data

    with open(id_maps_json) as f:
        id_maps = json.load(f)
    user2id: dict[str, int] = id_maps["user2id"]
    item2id: dict[str, int] = id_maps["item2id"]

    review_idx = _build_review_index(reviews_jsonl, user2id, item2id)

    n_ok = n_null = n_fail = n_mask = n_no_review = 0
    limit = dry_run if dry_run > 0 else max_records

    with open(out_path, "w", encoding="utf-8") as fout:
        for uid_str, entry in splits.items():
            if n_ok + n_null + n_fail + n_mask + n_no_review >= limit:
                break
            if not isinstance(entry, dict):
                continue
            test_item = entry.get("test")
            if test_item is None:
                continue

            uid = int(uid_str)
            user_reviews = review_idx.get(uid, {})
            pair = user_reviews.get(test_item)

            if pair is None:
                n_no_review += 1
                # Write null record so eval coverage is trackable
                fout.write(json.dumps({
                    "user_id": uid, "source": "eval", "memory_id": None,
                    "target_item": test_item, "query_text": None,
                    "src_review_ids": [], "masking_passed": False,
                    "failed_reason": "no_review",
                }, ensure_ascii=False) + "\n")
                continue

            review_text, item_title = pair
            prompt = (prompt_tmpl
                      .replace("{title}", item_title)
                      .replace("{review_text}", review_text[:1000]))

            query, reason = _llm_call(client, prompt)

            masking_passed = True
            if query and item_title and reason is None:
                masking_passed = _masking_passed(query, item_title)
                if not masking_passed:
                    reason = "mask_fail"
                    # retry once on mask fail
                    query2, reason2 = _llm_call(client, prompt, retry_max=1)
                    if query2 and _masking_passed(query2, item_title):
                        query, reason, masking_passed = query2, None, True
                    else:
                        n_mask += 1

            if reason == "null_query":
                n_null += 1
            elif reason == "llm_fail":
                n_fail += 1
            elif reason == "mask_fail":
                pass  # already counted
            else:
                n_ok += 1

            rec = {
                "user_id": uid,
                "source": "eval",
                "memory_id": None,
                "target_item": test_item,
                "query_text": query,
                "src_review_ids": [test_item],
                "masking_passed": masking_passed,
                "failed_reason": reason,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total = n_ok + n_null + n_fail + n_mask + n_no_review
            if total % 500 == 0 and total > 0:
                print(f"  [eval {total}] ok={n_ok} null={n_null} fail={n_fail} "
                      f"mask={n_mask} no_review={n_no_review}", flush=True)

    total = n_ok + n_null + n_fail + n_mask + n_no_review
    print(f"[p2_eval] done: total={total} ok={n_ok} null={n_null} fail={n_fail} "
          f"mask={n_mask} no_review={n_no_review}  pass_rate={n_ok/max(total,1):.3f}")


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)

    # train args
    parser.add_argument("--bank_dir", type=str, default="data/processed/Books/memory_bank")
    parser.add_argument("--candidates_jsonl", type=str,
                        default="data/processed/Books/books_memory_candidates.jsonl")
    parser.add_argument("--prompt_train", type=str,
                        default="config/prompts/amazon/Books/p2_pseudo_query_train.txt")

    # eval args
    parser.add_argument("--splits_json", type=str,
                        default="data/processed/Books/splits.json")
    parser.add_argument("--id_maps_json", type=str,
                        default="data/processed/Books/id_maps.json")
    parser.add_argument("--reviews_jsonl", type=str,
                        default="data/raw/Books/reviews.jsonl")
    parser.add_argument("--prompt_eval", type=str,
                        default="config/prompts/amazon/Books/p2_pseudo_query_eval.txt")

    # common
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--llm_config", type=str, default="configs/llm/p2.yaml")
    parser.add_argument("--max_records", type=int, default=999999)
    parser.add_argument("--dry_run", type=int, default=0,
                        help="If >0, process only this many records then exit")

    cli = parser.parse_args()

    from src.llm.client import load_llm_config, LLMClient
    llm_cfg = load_llm_config(cli.llm_config)
    client = LLMClient(llm_cfg)

    out_path = Path(cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cli.mode == "train":
        prompt_tmpl = Path(cli.prompt_train).read_text(encoding="utf-8")
        print(f"[p2_train] prompt: {cli.prompt_train}")
        print(f"[p2_train] bank: {cli.bank_dir}  output: {cli.output}")
        if cli.dry_run > 0:
            print(f"[p2_train] *** DRY RUN: {cli.dry_run} records ***")
        run_train(client, cli.candidates_jsonl, cli.bank_dir, prompt_tmpl, out_path,
                  cli.max_records, cli.dry_run)
    else:
        prompt_tmpl = Path(cli.prompt_eval).read_text(encoding="utf-8")
        print(f"[p2_eval] prompt: {cli.prompt_eval}")
        print(f"[p2_eval] splits: {cli.splits_json}  output: {cli.output}")
        if cli.dry_run > 0:
            print(f"[p2_eval] *** DRY RUN: {cli.dry_run} records ***")
        run_eval(client, cli.splits_json, cli.id_maps_json, cli.reviews_jsonl,
                 prompt_tmpl, out_path, cli.max_records, cli.dry_run)

    client.close()


if __name__ == "__main__":
    main()

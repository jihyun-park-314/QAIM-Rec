"""Throughput sweep: measure calls/hour at concurrency 1, 4, 8, 16.

Sends N_PER_LEVEL cold requests to Ollama (no SQLite cache) for each
concurrency level and records per-call latency + total wall-time.

Uses the fixed p1_books_b prompt and gemma4:26b.
Bypasses LLMClient cache so all calls hit Ollama.
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.prompts.p1_books import build_messages_b, P1_BOOKS_JSON_SCHEMA

# ---- config ----------------------------------------------------------------
LLM_CONFIG = "configs/llm/p1.yaml"
CANDIDATES = "data/processed/Books/books_memory_candidates.jsonl"
N_PER_LEVEL = 50          # cold calls per concurrency level
CONCURRENCIES = [1, 4, 8, 16]
SEED = 123                # different from smoke seed to avoid cached prompts
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_samples(n_total: int, seed: int) -> list[dict]:
    """Load n_total samples from candidates, skip first 100 (smoke set)."""
    rng = random.Random(seed)
    items = []
    with open(CANDIDATES) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    # skip first 100 used in smoke
    pool = items[100:]
    return rng.sample(pool, min(n_total, len(pool)))


def single_call(cfg: dict, item: dict) -> float:
    """Make one Ollama call, return latency_s. Returns -1.0 on failure."""
    messages, _ = build_messages_b(item)
    payload = {
        "model": cfg["model_id"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": int(cfg.get("max_new_tokens", 512))},
        "think": False,
        "format": P1_BOOKS_JSON_SCHEMA,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(cfg["api_url"], json=payload, timeout=120)
        resp.raise_for_status()
        return time.perf_counter() - t0
    except Exception as e:
        print(f"    [ERROR] {e}", flush=True)
        return -1.0


def run_level(cfg: dict, samples: list[dict], concurrency: int) -> dict:
    latencies = []
    t_wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(single_call, cfg, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            lat = fut.result()
            if lat > 0:
                latencies.append(lat)
    wall_s = time.perf_counter() - t_wall_start

    n = len(samples)
    n_ok = len(latencies)
    mean_lat = sum(latencies) / n_ok if n_ok else 0
    throughput_per_hour = n_ok / wall_s * 3600 if wall_s > 0 else 0

    return {
        "concurrency": concurrency,
        "n_calls": n,
        "n_ok": n_ok,
        "wall_s": round(wall_s, 1),
        "mean_latency_s": round(mean_lat, 2),
        "throughput_calls_per_hour": round(throughput_per_hour, 0),
    }


def main():
    cfg = load_config(LLM_CONFIG)
    n_total = N_PER_LEVEL * len(CONCURRENCIES)
    all_samples = load_samples(n_total, SEED)
    print(f"Loaded {len(all_samples)} cold samples (skipped first 100 smoke set)")
    print(f"Levels: {CONCURRENCIES}  x  {N_PER_LEVEL} calls each\n")

    results = []
    for i, c in enumerate(CONCURRENCIES):
        chunk = all_samples[i * N_PER_LEVEL: (i + 1) * N_PER_LEVEL]
        print(f"=== concurrency={c} ({N_PER_LEVEL} calls) ===", flush=True)
        r = run_level(cfg, chunk, c)
        results.append(r)
        print(f"  wall={r['wall_s']}s  mean_lat={r['mean_latency_s']}s  "
              f"throughput={r['throughput_calls_per_hour']:.0f} calls/hr\n", flush=True)

    print("\n=== THROUGHPUT SUMMARY ===")
    print(f"{'Concurrency':>12}  {'Calls/hr':>10}  {'Mean lat (s)':>13}  {'Wall (s)':>9}")
    print("-" * 50)
    baseline = results[0]["throughput_calls_per_hour"] if results else 1
    for r in results:
        speedup = r["throughput_calls_per_hour"] / baseline if baseline else 0
        print(f"{r['concurrency']:>12}  {r['throughput_calls_per_hour']:>10.0f}  "
              f"{r['mean_latency_s']:>13.2f}  {r['wall_s']:>9.1f}s  "
              f"(x{speedup:.1f} vs c=1)")

    # Saturation analysis
    if len(results) >= 2:
        gains = [results[i+1]["throughput_calls_per_hour"] / results[i]["throughput_calls_per_hour"]
                 for i in range(len(results)-1)]
        sat_idx = next((i for i, g in enumerate(gains) if g < 1.15), len(gains))
        sat_c = CONCURRENCIES[sat_idx] if sat_idx < len(CONCURRENCIES) else CONCURRENCIES[-1]
        print(f"\nSaturation point: concurrency={sat_c} "
              f"(next level gains <15%)")

    # Wall-time estimate for full extraction
    print("\n=== FULL EXTRACTION ESTIMATES (cap=12, 48,243 calls) ===")
    for r in results:
        if r["throughput_calls_per_hour"] > 0:
            hours = 48243 / r["throughput_calls_per_hour"]
            print(f"  c={r['concurrency']:>2}: {hours:.1f}h")

    out = "results/pilot/concurrency_sweep.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()

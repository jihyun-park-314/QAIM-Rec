"""
Prototype mega-cluster diagnosis: anisotropy check + linkage/tau sweep.
plan.md v0.4.8 §2.2(B)

LLM-free. Read-only on f3_bank.jsonl. No writes except final report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
BANK_PATH = ROOT / "data/memory/Books/f3_bank.jsonl"

# ── STEP 1: anisotropy check ──────────────────────────────────────────────────

def load_embeddings(path: Path) -> np.ndarray:
    vecs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                u = json.loads(line)
                vecs.append(u["embedding"]["vector"])
    return np.array(vecs, dtype=np.float32)


def mean_pairwise_cosine(vecs: np.ndarray, n_sample: int = 2000, seed: int = 42) -> float:
    """Sample n_sample random pairs, return mean cosine similarity."""
    rng = np.random.default_rng(seed)
    N = len(vecs)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vn = vecs / np.maximum(norms, 1e-9)

    i_idx = rng.integers(0, N, size=n_sample)
    j_idx = rng.integers(0, N, size=n_sample)
    # avoid self-pairs
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    sims = (vn[i_idx] * vn[j_idx]).sum(axis=1)
    return float(sims.mean()), float(sims.std())


# ── STEP 2: linkage sweep ─────────────────────────────────────────────────────

# sim thresholds to test → converted to euclidean distance (L2-norm assumed ~1)
# tau_euc = sqrt(2*(1 - sim))
SIM_THRESHOLDS = [0.75, 0.80, 0.85, 0.90]  # cosine sim thresholds
LINKAGES = ["complete", "ward"]


def sim_to_tau_euc(sim: float) -> float:
    return float((2.0 * (1.0 - sim)) ** 0.5)


SWEEP_SAMPLE = 5000  # subsample for sweep (5000×5000×4 ≈ 100MB — feasible)


def run_clustering(vecs_normed: np.ndarray, tau_euc: float, linkage: str,
                   rng: np.random.Generator) -> dict:
    """Run agglomerative on a SWEEP_SAMPLE subsample, return stats dict."""
    N_full = len(vecs_normed)
    if N_full > SWEEP_SAMPLE:
        idx = rng.choice(N_full, size=SWEEP_SAMPLE, replace=False)
        vecs = vecs_normed[idx]
    else:
        vecs = vecs_normed
    N = len(vecs)

    t0 = time.time()
    from sklearn.cluster import AgglomerativeClustering
    # sklearn 1.0.x uses 'affinity'; >=1.2 renamed it to 'metric'
    try:
        clf = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tau_euc,
            affinity="euclidean",
            linkage=linkage,
        )
    except TypeError:
        clf = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tau_euc,
            metric="euclidean",
            linkage=linkage,
        )
    labels = clf.fit_predict(vecs)
    elapsed = time.time() - t0

    label_ids, counts = np.unique(labels, return_counts=True)
    P = len(label_ids)
    max_size = int(counts.max())
    max_pct = max_size / N * 100
    ge20 = int((counts >= 20).sum())

    return {
        "linkage": linkage,
        "sim": None,  # filled by caller
        "tau_euc": round(tau_euc, 4),
        "P": P,
        "max_size": max_size,
        "max_pct": round(max_pct, 1),
        "ge20_clusters": ge20,
        "coverage_pct": 100.0,
        "N_used": N,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    print(f"Loading {BANK_PATH} ...")
    vecs = load_embeddings(BANK_PATH)
    N = len(vecs)
    print(f"N={N} vectors, dim={vecs.shape[1]}")

    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vn = vecs / np.maximum(norms, 1e-9)
    print(f"norm stats: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")

    # ── STEP 1 ──
    print("\n=== STEP 1: Anisotropy check ===")
    mean_sim, std_sim = mean_pairwise_cosine(vn, n_sample=4000)
    print(f"Mean pairwise cosine sim (n=4000 pairs): {mean_sim:.4f} ± {std_sim:.4f}")
    if mean_sim >= 0.70:
        print("  → SEVERE anisotropy (mean_sim ≥ 0.70). τ=0.35(=sim0.65) is below baseline — explains mega-cluster.")
    elif mean_sim >= 0.60:
        print("  → MODERATE anisotropy (0.60 ≤ mean_sim < 0.70). τ=0.35 is near baseline — linkage fix may help.")
    else:
        print("  → LOW anisotropy (mean_sim < 0.60). τ=0.35 should be fine — check other causes.")

    # ── STEP 2 ──
    print("\n=== STEP 2: Linkage + tau sweep ===")
    print(f"{'linkage':12s} {'sim':6s} {'tau_euc':8s} {'P':6s} {'max%':8s} {'≥20cls':7s} {'cov%':6s} {'sec':5s}")
    print("-" * 70)

    rng_sweep = np.random.default_rng(42)
    results = []
    for linkage in LINKAGES:
        for sim in SIM_THRESHOLDS:
            tau_euc = sim_to_tau_euc(sim)
            print(f"  Running {linkage} sim={sim} (N={min(len(vn), SWEEP_SAMPLE)})...", end=" ", flush=True)
            try:
                r = run_clustering(vn, tau_euc, linkage, rng_sweep)
                r["sim"] = sim
                results.append(r)
                print(f"done  P={r['P']:6d} max={r['max_pct']:5.1f}% ≥20={r['ge20_clusters']:3d}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

    print("\n── Full sweep table ──")
    print(f"{'linkage':12s} {'sim':6s} {'tau_euc':8s} {'P':6s} {'max%':8s} {'≥20cls':7s} {'cov%':6s}")
    for r in results:
        balanced = "✓" if r["max_pct"] < 40 and r["ge20_clusters"] >= 5 else " "
        print(f"{r['linkage']:12s} {r['sim']:.2f}   {r['tau_euc']:.4f}   {r['P']:6d}   {r['max_pct']:5.1f}%   {r['ge20_clusters']:4d}   {r['coverage_pct']:.1f}%  {balanced}")

    # ── STEP 3: Decision ──
    print("\n=== STEP 3: Decision ===")
    balanced_configs = [r for r in results if r["max_pct"] < 40 and r["ge20_clusters"] >= 5]

    # Also report top-P_MAX=15 stats for each config (prototype taxonomy view)
    P_MAX_PROTO = 15
    print(f"\nTop-{P_MAX_PROTO} cluster view (actual prototype count per plan §2.2(B) p_max={P_MAX_PROTO}):")
    print(f"  (All configs produce P_total >> {P_MAX_PROTO} — top-{P_MAX_PROTO} will all have ≥20 members if ≥20-member count ≥ {P_MAX_PROTO})")
    for r in results:
        note = "top-15 all ≥20 ✓" if r["ge20_clusters"] >= P_MAX_PROTO else f"only {r['ge20_clusters']} ≥20 clusters"
        print(f"  {r['linkage']:8s} sim={r['sim']:.2f}: P_total={r['P']:4d}, ≥20-clusters={r['ge20_clusters']:3d} → {note}")

    print()
    if balanced_configs:
        # For prototype use, prefer: (1) complete linkage (interpretable, no space distortion),
        # (2) sim=0.75 (most clusters, most ≥20 groups, more coverage)
        # Filter to those with ≥P_MAX_PROTO clusters having ≥20 members
        usable = [r for r in balanced_configs if r["ge20_clusters"] >= P_MAX_PROTO]
        if not usable:
            usable = balanced_configs
        best = sorted(usable, key=lambda r: (-r["ge20_clusters"], r["max_pct"]))[0]
        tau_code = 1.0 - best["sim"]  # cosine-distance space (code's τ convention)
        print(f"(a) BALANCED config found.")
        print(f"    Adopted: linkage={best['linkage']} sim={best['sim']} τ_euc={best['tau_euc']:.4f} (τ_code={tau_code:.2f})")
        print(f"    Cluster stats (N=5000 sample): P_total={best['P']}, max={best['max_pct']}%, ≥20-clusters={best['ge20_clusters']}")
        print(f"    Prototype synth: top-{P_MAX_PROTO} clusters → {P_MAX_PROTO} prototypes × ≤2 LLM calls = ≤{P_MAX_PROTO*2} calls.")
        print(f"    K=0 users (861): fallback reassemble via purchase-history centroid → nearest prototype.")
        print(f"    Personal bank (f3_bank.jsonl): UNCHANGED.")
        print(f"")
        print(f"    To apply: update build_prototypes.py LINKAGE='complete'→'{best['linkage']}', TAU_FALLBACKS=[{tau_code:.2f},...]")

        # Reproducibility check: run clustering twice on same subsample, compare labels
        print(f"\n── Reproducibility check (clustering only, LLM-free) ──")
        from sklearn.cluster import AgglomerativeClustering
        tau_euc = best["tau_euc"]
        linkage_name = best["linkage"]
        rng_r = np.random.default_rng(42)
        idx = rng_r.choice(len(vn), size=SWEEP_SAMPLE, replace=False)
        sub = vn[idx]
        run1_labels = None
        run2_labels = None
        for run_i, rng_seed in enumerate([42, 42]):
            try:
                clf = AgglomerativeClustering(n_clusters=None, distance_threshold=tau_euc,
                                              affinity="euclidean", linkage=linkage_name)
            except TypeError:
                clf = AgglomerativeClustering(n_clusters=None, distance_threshold=tau_euc,
                                              metric="euclidean", linkage=linkage_name)
            lab = clf.fit_predict(sub)
            if run_i == 0:
                run1_labels = lab
            else:
                run2_labels = lab
        match = np.all(run1_labels == run2_labels)
        print(f"  Run1 vs Run2 label match: {match} (deterministic sklearn agglomerative: expected True)")
        print(f"  Reproducibility of clustering step: {'PASS' if match else 'FAIL'}")
        print(f"  (Full _prototypes.json md5 reproducibility requires LLM synthesis with temp=0 cache — verify post-build)")
    else:
        print("(b) No balanced config found across all (linkage, τ) combinations.")
        print(f"    All configs either P too small or mega-cluster persists.")
        print(f"    Mean cosine sim = {mean_sim:.4f} → pure anisotropy: no geometric cluster structure.")
        print(f"    Descope options:")
        print(f"      · Single global prototype (= [DEFAULT_INTENT] equivalent for K=0 users — 'not personalized' explicitly).")
        print(f"      · Fixed K=8 spherical k-means (L2-normalized) — forced balanced 8 clusters, but no semantic structure guarantee.")
        print(f"    Recommendation: single global prototype or fixed-K=8 kmeans (state explicitly: structure absent).")

    print(f"\nPersonal bank (f3_bank.jsonl) unchanged. Diagnosis complete.")


if __name__ == "__main__":
    main()

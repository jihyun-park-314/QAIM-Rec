"""M2 mini-pipeline: P1 extract → discriminative filter → embed → cluster.

Pipeline stages (plan.md v0.4.3):
  1. extract     — run P1 compact Books on one candidate record
  2. filter      — is_discriminative AND grounding != metadata_dominant
                   AND contextual_intent non-empty AND evidence_span >= 1
  3. embed       — source_text = contextual_intent + " " + preference_summary
                   (no title/author/series)
  4. cluster     — per-user agglomerative, k_min=1, k_max=5, threshold tau

Leakage check: detect item_title tokens in source_text (same logic as p2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Per-record result after P1 extraction

@dataclass
class P1BooksRecord:
    """Result from running P1 compact Books on one candidate record."""
    # Input fields
    user_id: Any
    item_id: Any
    item_title: str
    item_category: Any  # list or str
    review_text_original: str  # original, not truncated

    # P1 output (None if parse failed)
    contextual_intent: str = ""
    preference_summary: str = ""
    evidence_span: list = field(default_factory=list)
    is_discriminative: bool = False
    grounding_level: str = ""

    # Pipeline metadata
    variant: str = "A"
    parse_failed: bool = False
    eligible: bool = False    # passes all 4 filter conditions
    source_text: str = ""     # for embedding (ci + ps)
    leakage_detected: bool = False  # title tokens in source_text

    # LLM stats
    latency_s: float = 0.0
    cache_hit: bool = False
    retry_count: int = 0
    truncated_input: bool = False
    input_tokens_approx: int = 0  # rough char-based estimate


def _category_str(category) -> str:
    if isinstance(category, list):
        return " > ".join(str(c) for c in category)
    return str(category) if category else ""


# ---------------------------------------------------------------------------
# Filter

def is_eligible(rec: P1BooksRecord) -> bool:
    """Apply all 4 filter conditions (plan.md v0.4.3)."""
    if rec.parse_failed:
        return False
    if not rec.is_discriminative:
        return False
    if rec.grounding_level == "metadata_dominant":
        return False
    if not rec.contextual_intent.strip():
        return False
    if len(rec.evidence_span) < 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Source text construction + leakage check

_STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "for", "with", "in", "on", "at", "to",
    "of", "by", "from", "as", "is", "it", "this", "that", "my", "i", "me",
    "was", "are", "be", "been", "have", "has", "had", "will", "would", "can",
    "could", "should", "very", "also", "but", "not", "so", "if", "all", "its",
])


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]


def make_source_text(rec: P1BooksRecord) -> str:
    """Concatenate contextual_intent + preference_summary for embedding."""
    parts = [rec.contextual_intent.strip(), rec.preference_summary.strip()]
    return " ".join(p for p in parts if p)


def check_leakage(source_text: str, title: str, min_token_len: int = 4) -> bool:
    """Return True if any significant title token appears in source_text."""
    title_tokens = {
        t for t in _tokenize(title)
        if len(t) >= min_token_len and t not in _STOP_WORDS
    }
    if not title_tokens:
        return False
    st_lower = source_text.lower()
    return any(tok in st_lower for tok in title_tokens)


# ---------------------------------------------------------------------------
# Leakage detector v2 — distinctiveness-based (post-hoc sidecar, STEP 2)

# Genre/category words that commonly appear in both titles and legitimate intent
# expressions, causing false positives in the unigram detector.
_GENRE_WORDS: frozenset = frozenset([
    "mystery", "romance", "thriller", "horror", "fantasy", "fiction", "nonfiction",
    "novel", "biography", "memoir", "history", "historical", "science", "literary",
    "classic", "adventure", "comedy", "drama", "action", "suspense", "crime",
    "detective", "psychological", "paranormal", "supernatural", "dystopian",
    "young", "adult", "children", "picture", "graphic", "series", "story", "stories",
    "tale", "tales", "chronicles", "saga", "epic", "dark", "love", "life", "world",
    "man", "woman", "girl", "boy", "house", "night", "day", "time", "new", "old",
    "great", "little", "last", "first", "secret", "lost", "dead", "blood", "black",
    "white", "red", "blue", "summer", "winter", "spring", "autumn", "fall",
    "piano", "music", "war", "peace", "game", "fire", "water", "earth", "light",
    "shadow", "heart", "soul", "mind", "body", "death", "life", "dream", "hope",
    "fear", "power", "truth", "magic", "dragon", "sword", "king", "queen",
    "complete", "book", "guide", "complete", "edition", "volume", "part",
])


def build_common_tokens_from_titles(
    titles: list[str],
    df_threshold: float = 0.05,
) -> frozenset:
    """Build high-document-frequency token set from a corpus of titles.

    Tokens appearing in >= df_threshold fraction of titles are considered
    corpus-common and treated like genre words (not distinctive).
    """
    from collections import Counter
    n = len(titles)
    if n == 0:
        return frozenset()
    doc_freq: Counter = Counter()
    for title in titles:
        tokens = set(_tokenize(title))
        for tok in tokens:
            if tok not in _STOP_WORDS:
                doc_freq[tok] += 1
    threshold_count = max(1, int(df_threshold * n))
    return frozenset(tok for tok, cnt in doc_freq.items() if cnt >= threshold_count)


def check_leakage_v2(
    source_text: str,
    title: str,
    author: str = "",
    common_tokens: frozenset | None = None,
) -> bool:
    """Distinctiveness-based leakage detector (v2).

    Leakage = True if:
      (a) any consecutive 2+ distinctive title tokens appear as a phrase in
          source_text, OR
      (b) any author token (even single) appears in source_text.

    Distinctive tokens = title tokens minus stopwords, genre words, and
    corpus-common tokens (passed via common_tokens).

    A single genre/common word in the title matching source_text is NOT leakage.
    """
    extra_common = common_tokens or frozenset()
    exclude = _STOP_WORDS | _GENRE_WORDS | extra_common

    title_toks = _tokenize(title)
    distinctive = [t for t in title_toks if t not in exclude]

    st_lower = source_text.lower()

    # Check consecutive distinctive n-grams (n >= 2) as phrase match
    for n in range(len(distinctive), 1, -1):
        for start in range(len(distinctive) - n + 1):
            phrase = " ".join(distinctive[start:start + n])
            if phrase in st_lower:
                return True

    # Author check: any author token (even single) triggers leakage
    if author:
        author_toks = _tokenize(author)
        for tok in author_toks:
            if tok and tok not in _STOP_WORDS and tok in st_lower:
                # Require word-boundary match to avoid substring false positives
                if re.search(r"\b" + re.escape(tok) + r"\b", st_lower):
                    return True

    return False


def recompute_leakage_field(
    records: list[dict],
    author_map: dict | None = None,
    common_tokens: frozenset | None = None,
) -> tuple[list[dict], dict]:
    """Apply check_leakage_v2 post-hoc to a list of p1_extractions records.

    Returns (updated_records, stats) where stats reports old vs new fire rate.
    Does NOT modify extract_record() or the extraction pipeline.

    author_map: {item_id -> author_str} from meta.jsonl. Pass None to skip
                author-name leakage detection.
    common_tokens: output of build_common_tokens_from_titles(), or None.
    """
    old_true = 0
    new_true = 0
    flipped_off = 0
    flipped_on = 0
    updated = []
    for rec in records:
        old_val = bool(rec.get("leakage_detected", False))
        author = ""
        if author_map is not None:
            author = author_map.get(rec.get("item_id"), "") or ""
        new_val = check_leakage_v2(
            rec.get("source_text", ""),
            rec.get("item_title", ""),
            author=author,
            common_tokens=common_tokens,
        )
        if old_val:
            old_true += 1
        if new_val:
            new_true += 1
        if old_val and not new_val:
            flipped_off += 1
        if not old_val and new_val:
            flipped_on += 1
        updated.append({**rec, "leakage_detected": new_val})

    n = len(records)
    stats = {
        "n_records": n,
        "old_fire_rate": round(old_true / max(n, 1), 4),
        "new_fire_rate": round(new_true / max(n, 1), 4),
        "flipped_off": flipped_off,  # FP removed
        "flipped_on": flipped_on,    # previously missed, now caught
    }
    return updated, stats


# ---------------------------------------------------------------------------
# Extract + annotate one record

def extract_record(
    client,
    item: dict,
    variant: str = "A",
    retry_max: int = 2,
) -> P1BooksRecord:
    """Run P1 compact Books on one item and return an annotated P1BooksRecord.

    `item` is a row from books_memory_candidates.jsonl with keys:
    user_id, item_id, item_title, item_category, review_text, item_description, rating.
    """
    from src.llm.prompts.p1_books import run_p1_books

    result = run_p1_books(client, item, variant=variant, retry_max=retry_max)

    rec = P1BooksRecord(
        user_id=item["user_id"],
        item_id=item["item_id"],
        item_title=item.get("item_title", ""),
        item_category=item.get("item_category", ""),
        review_text_original=item.get("review_text", ""),
        variant=variant,
        latency_s=result.latency_s,
        cache_hit=result.cache_hit,
        retry_count=result.retry_count,
        truncated_input=result.truncated_input,
    )

    # Approximate input tokens (rough: chars / 4)
    rec.input_tokens_approx = len(item.get("review_text", "")) // 4

    if result.parsed is None:
        rec.parse_failed = True
        rec.eligible = False
        return rec

    p = result.parsed
    rec.contextual_intent = (p.get("contextual_intent") or "").strip()
    rec.preference_summary = (p.get("preference_summary") or "").strip()
    rec.evidence_span = p.get("evidence_span") or []
    rec.is_discriminative = bool(p.get("is_discriminative", False))
    rec.grounding_level = p.get("grounding_level", "")

    rec.eligible = is_eligible(rec)
    rec.source_text = make_source_text(rec)
    rec.leakage_detected = check_leakage(rec.source_text, rec.item_title)

    return rec


# ---------------------------------------------------------------------------
# Clustering

def cluster_user_records(
    eligible_records: list[P1BooksRecord],
    emb_model,
    k_min: int = 1,
    k_max: int = 5,
    tau: float = 0.3,
) -> dict:
    """Cluster eligible records for one user.

    Returns:
      {
        "k_personal": int,
        "labels": list[int],        # cluster label per eligible record
        "embeddings": np.ndarray,   # [n, d]
        "cluster_summaries": [
          {"label": int, "size": int, "source_texts": [str], "intents": [str]}
        ]
      }
    """
    n = len(eligible_records)
    if n == 0:
        return {"k_personal": 0, "labels": [], "embeddings": None, "cluster_summaries": []}

    source_texts = [r.source_text for r in eligible_records]
    vecs = emb_model.encode_corpus(source_texts)  # [n, d], L2-normalized

    if n == 1:
        labels = [0]
    else:
        labels = _agglomerative(vecs, tau, k_min, k_max)

    k = len(set(labels))
    summaries = []
    for c in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == c]
        summaries.append({
            "label": c,
            "size": len(idxs),
            "source_texts": [eligible_records[i].source_text for i in idxs],
            "intents": [eligible_records[i].contextual_intent for i in idxs],
            "evidence_spans": [eligible_records[i].evidence_span for i in idxs],
        })

    return {
        "k_personal": k,
        "labels": labels,
        "embeddings": vecs,
        "cluster_summaries": summaries,
    }


def _agglomerative(
    vecs: np.ndarray,
    tau: float,
    k_min: int,
    k_max: int,
) -> list[int]:
    """Agglomerative clustering on L2-normalized vectors.

    Uses average linkage with cosine distance (= 1 - cosine_sim for unit vectors).
    Threshold tau; caps at k_max; k_min=1 is naturally satisfied.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(vecs)
    if n <= 1:
        return [0] * n

    # Average linkage + cosine metric (sklearn expects precomputed for cosine)
    # vecs are already L2-normalized; use euclidean on normalized ~ cosine distance
    # (euclidean on unit vecs: d = sqrt(2*(1-cos)) — monotone with cosine distance)
    # We use distance_threshold in euclidean space; tau_euc = sqrt(2 * tau_cos)
    tau_euc = (2.0 * tau) ** 0.5

    clf = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tau_euc,
        metric="euclidean",
        linkage="average",
    )
    labels_arr = clf.fit_predict(vecs)
    k = len(set(labels_arr))

    if k > k_max:
        clf2 = AgglomerativeClustering(n_clusters=k_max, metric="euclidean", linkage="average")
        labels_arr = clf2.fit_predict(vecs)

    return labels_arr.tolist()

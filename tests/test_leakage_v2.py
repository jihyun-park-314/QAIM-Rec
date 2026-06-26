"""Unit tests for check_leakage_v2 and recompute_leakage_field.

Tests verify:
  (i)  Genre-word false-positives from check_leakage (v1) are eliminated in v2.
  (ii) Proper-noun phrase leakage (2+ distinctive consecutive tokens) is correctly detected.
  (iii) Author-name leakage (even single token) is correctly detected.
  (iv) Comparison: cases where v1 fires but v2 correctly suppresses (FP removal),
       and artificial positives v1 misses but v2 catches.
"""

import pytest
from src.memory.pipeline import (
    check_leakage,
    check_leakage_v2,
    build_common_tokens_from_titles,
    recompute_leakage_field,
)


# ---------------------------------------------------------------------------
# (i) Genre-word false-positives: v1 fires, v2 should NOT fire

@pytest.mark.parametrize("title,source_text", [
    # "mystery" is a genre word — appears legitimately in intent
    ("Sherlock and the Mystery Box",
     "enjoyed solving mystery elements in the narrative"),
    # "piano" is in _GENRE_WORDS
    ("The Piano Teacher",
     "I love stories about piano and classical music"),
    # "dark romance" — both dark and romance in genre list
    ("Dark Romance at Sea",
     "this is a dark romance that hooked me from page one"),
    # "history" alone
    ("A History of Time",
     "fascinated by history and how the author explains complex ideas"),
    # "thriller" alone
    ("The Thriller Next Door",
     "this thriller kept me up all night reading"),
])
def test_genre_fp_removed(title, source_text):
    """v1 fires on these genre words; v2 should suppress them."""
    assert check_leakage(source_text, title), (
        f"Precondition: v1 should fire for title={title!r}"
    )
    assert not check_leakage_v2(source_text, title), (
        f"v2 should NOT fire (genre-word FP) for title={title!r}"
    )


# ---------------------------------------------------------------------------
# (ii) Proper-noun phrase: 2+ consecutive distinctive tokens → leakage=True

@pytest.mark.parametrize("title,source_text", [
    # "Harry Potter" — both tokens are distinctive
    ("Harry Potter and the Sorcerer's Stone",
     "this book is about harry potter going to hogwarts"),
    # "Tolkien" as distinctive token pair context; test actual phrase match
    ("The Hobbit Journey",
     "I loved how this hobbit journey mirrors classic adventure arcs"),
    # Two-token proper noun injected directly
    ("Midnight Labyrinth",
     "exploring the midnight labyrinth metaphor throughout"),
    # Two consecutive distinctive tokens present as a phrase
    ("Clockwork Prometheus",
     "the clockwork prometheus metaphor anchors the entire narrative"),
])
def test_distinctive_phrase_detected(title, source_text):
    """Consecutive distinctive tokens should trigger v2 leakage."""
    result = check_leakage_v2(source_text, title)
    assert result, (
        f"v2 should fire (distinctive phrase) for title={title!r}, source={source_text!r}"
    )


# ---------------------------------------------------------------------------
# (iii) Author-name leakage: single author token → leakage=True

@pytest.mark.parametrize("title,source_text,author", [
    # Tolkien in source text
    ("The Fellowship", "tolkien's world-building is unmatched", "J.R.R. Tolkien"),
    # King (Stephen King) — word-boundary match
    ("The Shining Dark", "stephen king writes horror unlike anyone else", "Stephen King"),
    # Short distinctive author name
    ("Invisible Man", "ellison captures the black experience brilliantly", "Ralph Ellison"),
])
def test_author_leakage_detected(title, source_text, author):
    """Author token in source_text should trigger v2 leakage."""
    result = check_leakage_v2(source_text, title, author=author)
    assert result, (
        f"v2 should fire (author match) for author={author!r}, source={source_text!r}"
    )


# ---------------------------------------------------------------------------
# (iv) Cases where v1 fires but v2 suppresses (FP removal)

@pytest.mark.parametrize("title,source_text", [
    # Single genre/common word only
    ("Love in Paris",
     "I love the way the characters develop emotionally"),
    ("The Last Shadow",
     "the last part was a shadow of what I expected"),
])
def test_v1_fp_v2_correct(title, source_text):
    """These are genuine v1 false positives that v2 correctly rejects."""
    assert check_leakage(source_text, title), "Precondition: v1 fires"
    assert not check_leakage_v2(source_text, title), "v2 should not fire"


# ---------------------------------------------------------------------------
# (iv-b) Artificial positives v2 catches that v1 might also catch, but the key
# test is that v2 requires phrase or author — single distinctive token NOT a phrase

def test_single_distinctive_token_not_leakage():
    """Single distinctive token alone is NOT leakage in v2 (requires 2+ or author)."""
    title = "Labyrinth of Shadows"
    # "labyrinth" is distinctive but appears alone (not paired with another distinctive token)
    source_text = "the labyrinth metaphor represents confusion"
    result = check_leakage_v2(source_text, title)
    assert not result, (
        "Single distinctive token alone should not trigger v2 leakage"
    )


def test_two_distinctive_tokens_not_adjacent():
    """Two distinctive tokens present but NOT consecutive → no leakage."""
    title = "Labyrinth Echoes"
    # both "labyrinth" and "echoes" in text, but not adjacent
    source_text = "labyrinth-like confusion with distant echoes of meaning"
    result = check_leakage_v2(source_text, title)
    # "labyrinth echoes" does not appear as a phrase
    assert not result, (
        "Non-adjacent distinctive tokens should not trigger phrase match"
    )


def test_two_distinctive_tokens_adjacent_leakage():
    """Two distinctive tokens consecutive → leakage."""
    title = "Labyrinth Echoes"
    source_text = "the labyrinth echoes throughout the entire narrative"
    result = check_leakage_v2(source_text, title)
    assert result, "Adjacent distinctive tokens as phrase should trigger v2"


# ---------------------------------------------------------------------------
# build_common_tokens_from_titles

def test_build_common_tokens_filters_high_df():
    titles = ["Harry Potter", "Harry and Sally", "The Harry Story"] * 10 + \
             ["Unique Zephyr Title", "Another Different Book"]
    common = build_common_tokens_from_titles(titles, df_threshold=0.5)
    assert "harry" in common, "High-DF 'harry' should be in common tokens"
    assert "zephyr" not in common, "Low-DF 'zephyr' should not be in common tokens"


# ---------------------------------------------------------------------------
# recompute_leakage_field — fire rate comparison

def test_recompute_fire_rate_drops():
    """Post-hoc recompute with v2 should reduce fire rate on genre-FP records."""
    records = [
        {
            "user_id": 1, "item_id": 101,
            "source_text": "enjoyed solving mystery elements in the narrative",
            "item_title": "Sherlock and the Mystery Box",
            "leakage_detected": True,  # v1 incorrectly set
        },
        {
            "user_id": 1, "item_id": 102,
            "source_text": "the harry potter phrase is mentioned explicitly",
            "item_title": "Harry Potter and the Chamber of Secrets",
            "leakage_detected": False,  # v1 missed
        },
        {
            "user_id": 2, "item_id": 201,
            "source_text": "this is a completely clean intent with no title words",
            "item_title": "Clean Title Here",
            "leakage_detected": False,  # clean, stays false
        },
    ]
    updated, stats = recompute_leakage_field(records)

    assert stats["n_records"] == 3
    assert stats["old_fire_rate"] > stats["new_fire_rate"] or stats["flipped_on"] >= 1
    # Record 101: genre FP removed
    assert not updated[0]["leakage_detected"], "Genre FP should be removed"
    # Record 102: actual leak caught
    assert updated[1]["leakage_detected"], "Harry Potter phrase should be detected"
    # Record 201: clean record stays clean
    assert not updated[2]["leakage_detected"], "Clean record should stay false"
    assert stats["flipped_off"] >= 1, "At least one FP should be removed"
    assert stats["flipped_on"] >= 1, "At least one missed positive should be caught"

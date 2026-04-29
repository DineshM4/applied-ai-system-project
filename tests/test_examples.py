"""
Living walkthrough of all three system paths.

Run with:  pytest -s tests/test_examples.py

Each test prints a labelled block showing the query, any guardrail warnings
that fired, the ranked songs returned, and the explanation — so the output
itself is the proof of which path was taken.
"""

import logging
import pytest
from unittest.mock import patch

from src.recommender import load_songs, build_chroma_collection, rag_recommend

THICK = "═" * 64
THIN  = "─" * 64

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def catalog():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def collection(catalog):
    return build_chroma_collection(catalog)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(path_label: str, description: str, query: str) -> None:
    print(f"\n{THICK}")
    print(f"  {path_label}")
    print(f"  {description}")
    print(f"  query : \"{query}\"")
    print(THICK)


def _print_warnings(messages: list) -> None:
    if messages:
        print(f"\n  guardrail warnings fired:")
        for m in messages:
            print(f"    ⚠  {m}")
    else:
        print(f"\n  guardrail warnings : none")


def _print_songs(songs: list) -> None:
    print(f"\n  ranked results:")
    for i, s in enumerate(songs, 1):
        print(f"  #{i}  {s['title']} — {s['artist']}")
        print(f"       genre={s['genre']}  mood={s['mood']}"
              f"  energy={s['energy']:.2f}  bpm={s['tempo_bpm']:.0f}")


def _print_explanation(text: str, max_lines: int = 6) -> None:
    print(f"\n  explanation:")
    for line in text.strip().splitlines()[:max_lines]:
        print(f"    {line}")


# ── PATH 1: Full RAG happy path ───────────────────────────────────────────────

_HAPPY_PROFILE = {
    "favorite_genre": ["lofi", "ambient"],
    "favorite_mood":  "chill",
    "target_energy":  0.35,
    "target_acousticness": 0.75,
}
_HAPPY_EXPLANATION = (
    "Here's your perfect study session playlist!\n"
    "Library Rain by Paper Lanterns sets a tranquil, rain-soaked mood — ideal for deep focus.\n"
    "Midnight Coding by LoRoom brings lo-fi warmth at a slow 78 BPM groove.\n"
    "Spacewalk Thoughts by Orbit Bloom drifts in ambient textures for distraction-free listening."
)


def test_path1_full_rag_happy_path(catalog, collection, caplog):
    """
    PATH 1 — Both Gemini calls succeed.
    extract_profile_from_query returns a structured profile;
    generate_explanation returns a DJ-style summary.
    Expected: source='rag', no guardrail warnings.
    """
    QUERY = "chill acoustic study music"

    _print_header(
        "PATH 1 — Full RAG  (ChromaDB + Gemini profile + Gemini explanation)",
        "Both LLM calls succeed — no guardrails triggered.",
        QUERY,
    )

    with caplog.at_level(logging.WARNING, logger="root"):
        with patch("src.recommender.extract_profile_from_query",
                   return_value=_HAPPY_PROFILE):
            with patch("src.recommender.generate_explanation",
                       return_value=_HAPPY_EXPLANATION):
                result = rag_recommend(QUERY, collection, catalog, k=3)

    _print_warnings(caplog.messages)
    _print_songs(result["songs"])
    _print_explanation(result["explanation"])
    print(f"\n  source={result['source']}  ← full RAG pipeline completed")
    print(THICK)

    assert result["source"] == "rag"
    assert len(result["songs"]) == 3
    assert not caplog.messages, "Happy path must not fire any guardrail warnings"


# ── PATH 2: Inner guard — profile extraction fails ────────────────────────────

def test_path2_inner_guard_profile_extraction_fails(catalog, collection, caplog):
    """
    PATH 2 — Gemini profile extraction throws ConnectionError.
    Inner guard fires: [PROFILE EXTRACT] warning logged, soft_profile=None,
    rerank_candidates uses 100% semantic score to preserve query intent.
    Expected: source='rag', one [PROFILE EXTRACT] warning, query-relevant songs.
    """
    QUERY = "passionate dramatic orchestral music"

    _print_header(
        "PATH 2 — Inner Guard  (profile extraction throws → semantic-only ranking)",
        "extract_profile_from_query raises ConnectionError mid-pipeline.",
        QUERY,
    )

    with caplog.at_level(logging.WARNING, logger="root"):
        with patch("src.recommender.extract_profile_from_query",
                   side_effect=ConnectionError("Gemini API unreachable")):
            with patch("src.recommender.generate_explanation",
                       return_value="[explanation mocked — inner guard demo]"):
                result = rag_recommend(QUERY, collection, catalog, k=3)

    _print_warnings(caplog.messages)
    _print_songs(result["songs"])
    _print_explanation(result["explanation"])
    print(f"\n  source={result['source']}"
          f"  ← pipeline recovered; ChromaDB semantic scores drove ranking")
    print(THICK)

    assert result["source"] == "rag", "Inner guard must not escalate to outer fallback"
    assert len(result["songs"]) == 3
    assert any("[PROFILE EXTRACT]" in m for m in caplog.messages), \
        "Expected [PROFILE EXTRACT] warning as proof inner guard fired"


# ── PATH 3: Outer guard — full pipeline failure ───────────────────────────────

def test_path3_outer_guard_chromadb_failure(catalog, collection, caplog):
    """
    PATH 3 — ChromaDB raises RuntimeError before any Gemini call is made.
    Outer guard fires: [FALLBACK] warning logged, recommend_songs() runs
    against NEUTRAL_PROFILE using pure math scoring.
    Expected: source='fallback', one [FALLBACK] warning, query-unaware results.
    """
    QUERY = "aggressive high-energy metal"

    _print_header(
        "PATH 3 — Outer Guard  (ChromaDB down → pure math fallback)",
        "semantic_recommend raises RuntimeError; entire RAG pipeline aborts.",
        QUERY,
    )

    with caplog.at_level(logging.WARNING, logger="root"):
        with patch("src.recommender.semantic_recommend",
                   side_effect=RuntimeError("ChromaDB connection lost")):
            result = rag_recommend(QUERY, collection, catalog, k=3)

    _print_warnings(caplog.messages)
    _print_songs(result["songs"])
    _print_explanation(result["explanation"])
    print(f"\n  source={result['source']}"
          f"  ← query-unaware; NEUTRAL_PROFILE math returned regardless of query")
    print(THICK)

    assert result["source"] == "fallback"
    assert len(result["songs"]) == 3
    assert any("[FALLBACK]" in m for m in caplog.messages), \
        "Expected [FALLBACK] warning as proof outer guard fired"

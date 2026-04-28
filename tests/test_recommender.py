import pytest
from src.recommender import (
    Song, UserProfile, Recommender,
    load_songs, recommend_songs, build_chroma_collection, semantic_recommend,
)

# NOTE: ALL followings tests in this file is done for the deterministic part of the music reccomender. The LLM parts are done in main.py

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy",
            energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill",
            energy=0.4, tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ]
    return Recommender(songs)


USER_PROFILES = [
    {
        "name": "Chill Lofi Listener",
        "favorite_genre":      ["lofi", "pop"],
        "favorite_mood":       "chill",
        "target_energy":       0.4,
        "target_acousticness": 0.7,
    },
    {
        "name": "High-Energy Rock Fan",
        "favorite_genre":      ["rock", "metal"],
        "favorite_mood":       "energetic",
        "target_energy":       0.9,
        "target_acousticness": 0.1,
    },
    {
        "name": "Jazz & Soul Explorer",
        "favorite_genre":      ["jazz", "r&b"],
        "favorite_mood":       "relaxed",
        "target_energy":       0.35,
        "target_acousticness": 0.8,
    },
]

CONTRADICTORY_USER_PROFILES = [
    {
        "name": "Genre Vacuum",
        "favorite_genre":      [],
        "favorite_mood":       "chill",
        "target_energy":       0.4,
        "target_acousticness": 0.7,
    },
    {
        "name": "Family Maximizer",
        "favorite_genre":      ["lofi", "rock", "jazz", "hip-hop"],
        "favorite_mood":       "chill",
        "target_energy":       0.5,
        "target_acousticness": 0.5,
    },
    {
        "name": "Ghost Mood",
        "favorite_genre":      ["pop"],
        "favorite_mood":       "triumphant",
        "target_energy":       0.8,
        "target_acousticness": 0.2,
    },
    {
        "name": "Contradictory Listener",
        "favorite_genre":      ["classical"],
        "favorite_mood":       "aggressive",
        "target_energy":       0.95,
        "target_acousticness": 0.05,
    },
    {
        "name": "Mood Adjacent",
        "favorite_genre":      ["lofi"],
        "favorite_mood":       "chill",
        "target_energy":       0.4,
        "target_acousticness": 0.8,
    },
    {
        "name": "Valence Hijacker",
        "favorite_genre":      ["rock"],
        "favorite_mood":       "happy",
        "target_energy":       0.9,
        "target_acousticness": 0.1,
    },
    {
        "name": "Perfect Ringer (tuned to Library Rain)",
        "favorite_genre":      ["lofi"],
        "favorite_mood":       "chill",
        "target_energy":       0.35,
        "target_acousticness": 0.86,
    },
]

SEMANTIC_QUERIES = [
    "I want something chill and acoustic to study to",
    "Give me aggressive high-energy rock",
    "Upbeat electronic music for a workout",
]

# ── Recommender class (OOP interface) ─────────────────────────────────────────


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre=["pop"],
        favorite_mood="happy",
        target_energy=0.8,
        target_acousticness=0.2,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre=["pop"],
        favorite_mood="happy",
        target_energy=0.8,
        target_acousticness=0.2,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ── recommend_songs (functional interface, full catalog) ──────────────────────

@pytest.fixture(scope="module")
def catalog():
    return load_songs("data/songs.csv")


@pytest.mark.parametrize("user_prefs", USER_PROFILES, ids=[p["name"] for p in USER_PROFILES])
def test_recommend_songs_user_profiles(user_prefs, catalog):
    results = recommend_songs(user_prefs, catalog, k=5)

    assert len(results) == 5
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("user_prefs", CONTRADICTORY_USER_PROFILES, ids=[p["name"] for p in CONTRADICTORY_USER_PROFILES])
def test_recommend_songs_edge_cases(user_prefs, catalog):
    results = recommend_songs(user_prefs, catalog, k=5)

    assert len(results) == 5
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_perfect_ringer_returns_library_rain_first(catalog):
    perfect_ringer = next(
        p for p in CONTRADICTORY_USER_PROFILES if p["name"].startswith("Perfect Ringer"))
    results = recommend_songs(perfect_ringer, catalog, k=5)

    assert results[0][0]["title"] == "Library Rain"


# ── semantic_recommend (ChromaDB, no LLM) ─────────────────────────────────────

@pytest.fixture(scope="module")
def chroma_collection(catalog):
    return build_chroma_collection(catalog)


@pytest.mark.parametrize("query", SEMANTIC_QUERIES)
def test_semantic_recommend_returns_k_results(query, chroma_collection):
    results = semantic_recommend(query, chroma_collection, k=3)

    assert len(results) == 3
    for song_dict, distance, document in results:
        assert isinstance(song_dict["title"], str)
        assert distance >= 0
        assert isinstance(document, str)

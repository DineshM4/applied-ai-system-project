"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs, build_chroma_collection, rag_recommend

DIVIDER = "─" * 52
THIN_DIVIDER = "·" * 52

# Three distinct listener personas
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


def print_recommendations(user_prefs: dict, recommendations: list) -> None:
    """Print the header and ranked results for a single user profile."""
    print(f"\n{DIVIDER}")
    print(f"  MUSIC RECOMMENDER — {user_prefs['name']}")
    print(DIVIDER)
    print(f"  Genres   : {', '.join(user_prefs['favorite_genre'])}")
    print(f"  Mood     : {user_prefs['favorite_mood']}")
    print(f"  Energy   : {user_prefs['target_energy']}")
    print(f"  Acoustic : {user_prefs['target_acousticness']}")
    print(DIVIDER)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(
            f"       Score: {score:.4f}  |  Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"  {THIN_DIVIDER}")
        for reason in explanation.split(" | "):
            print(f"    • {reason}")

    print(f"\n{DIVIDER}\n")


SEMANTIC_QUERIES = [
    "I want something chill and acoustic to study to",
    "Give me aggressive high-energy rock",
    "Upbeat electronic music for a workout",
]

THICK_DIVIDER = "═" * 52


def print_rag_result(query: str, result: dict) -> None:
    """Print a single RAG query result in a structured, readable format."""
    source_label = f"[{result['source'].upper()}]"
    print(f"\n{DIVIDER}")
    print(f"  QUERY  {source_label}")
    print(f'  "{query}"')
    print(DIVIDER)

    for rank, song in enumerate(result["songs"], start=1):
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']:.2f}  |  BPM: {song['tempo_bpm']:.0f}")

    print(f"\n  {THIN_DIVIDER}")
    print("  Explanation:")
    for line in result["explanation"].strip().splitlines():
        print(f"    {line}")
    print(f"\n{DIVIDER}")


def run_semantic_demo() -> None:
    songs = load_songs("data/songs.csv")
    collection = build_chroma_collection(songs)

    print(f"\n{THICK_DIVIDER}")
    print("  SEMANTIC RAG RECOMMENDER")
    print(f"  {len(SEMANTIC_QUERIES)} queries  |  ChromaDB + Gemini")
    print(f"{THICK_DIVIDER}")

    for query in SEMANTIC_QUERIES:
        result = rag_recommend(query, collection, songs, k=5)
        print_rag_result(query, result)

    print(f"\n{THICK_DIVIDER}")
    print("  INTERACTIVE MODE  —  type your own query")
    print(f"  (enter 'q' or leave blank to quit)")
    print(f"{THICK_DIVIDER}\n")

    while True:
        try:
            query = input("  Your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() == "q":
            break
        result = rag_recommend(query, collection, songs, k=5)
        print_rag_result(query, result)

    print(f"\n{THICK_DIVIDER}\n")


def main() -> None:
    songs = load_songs("data/songs.csv")

    for user_prefs in CONTRADICTORY_USER_PROFILES:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(user_prefs, recommendations)

    for user_prefs in USER_PROFILES:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(user_prefs, recommendations)


if __name__ == "__main__":
    run_semantic_demo()
    main()

"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs

DIVIDER = "─" * 52
THIN_DIVIDER = "·" * 52


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {
        "favorite_genre":     ["lofi", "pop"],
        "favorite_mood":      "chill",
        "target_energy":      0.4,
        "target_acousticness": 0.7,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  MUSIC RECOMMENDER — Top Picks")
    print(DIVIDER)
    print(f"  Genres   : {', '.join(user_prefs['favorite_genre'])}")
    print(f"  Mood     : {user_prefs['favorite_mood']}")
    print(f"  Energy   : {user_prefs['target_energy']}")
    print(f"  Acoustic : {user_prefs['target_acousticness']}")
    print(DIVIDER)

    # ── Results ───────────────────────────────────────────────────────────────
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(
            f"       Score: {score:.4f}  |  Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"  {THIN_DIVIDER}")
        for reason in explanation.split(" | "):
            print(f"    • {reason}")

    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()

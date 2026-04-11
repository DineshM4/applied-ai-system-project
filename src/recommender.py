from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: list[str]
    favorite_mood: str
    target_energy: float
    target_acousticness: float


GENRE_FAMILIES = {
    "electronic": {"lofi", "synthwave", "electronic", "ambient"},
    "rock_guitar": {"rock", "metal", "indie pop", "folk", "country"},
    "vocal_urban": {"hip-hop", "r&b", "pop", "emo"},
    "acoustic_world": {"jazz", "classical", "reggae", "world"},
}


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        """Store the catalog of songs for use during recommendation."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the first k songs from the catalog (recommendation logic not yet implemented)."""
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a placeholder explanation string (explanation logic not yet implemented)."""
        # TODO: Implement explanation logic
        return "Explanation placeholder"


def load_songs(csv_path: str) -> List[Dict]:
    """Parse a CSV file into a list of song dicts with correctly typed numeric fields."""
    import csv

    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm",
                    "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song = {}
            for key, value in row.items():
                if key in int_fields:
                    song[key] = int(value)
                elif key in float_fields:
                    song[key] = float(value)
                else:
                    song[key] = value
            songs.append(song)
    return songs


MOOD_VALENCE_TARGET = {
    "euphoric":    0.90,
    "happy":       0.85,
    "romantic":    0.85,
    "energetic":   0.75,
    "laid-back":   0.75,
    "peaceful":    0.70,
    "relaxed":     0.70,
    "nostalgic":   0.62,
    "chill":       0.60,
    "focused":     0.57,
    "spiritual":   0.55,
    "moody":       0.48,
    "intense":     0.45,
    "melancholic": 0.35,
    "sad":         0.20,
    "aggressive":  0.20,
}


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    Returns (score, reasons) where score is 0.0–1.0 and reasons is a
    list of human-readable strings explaining each component.

    Weights:
      S1 Genre Match        0.25
      S2 Mood Match         0.20
      S3 Energy Proximity   0.30
      S4 Acoustic Pref      0.15
      S5 Valence Proximity  0.10
    """
    reasons: List[str] = []

    # ── S1: Genre Match (weight 0.25) ────────────────────────────────────────
    fav_genres = {g.lower() for g in user_prefs.get("favorite_genre", [])}
    song_genre = song.get("genre", "").lower()

    if song_genre in fav_genres:
        genre_score = 1.0
        reasons.append(f"genre '{song_genre}' is a favorite (full points)")
    else:
        song_family = next(
            (fam for fam, members in GENRE_FAMILIES.items() if song_genre in members),
            None,
        )
        user_families = {
            fam for fam, members in GENRE_FAMILIES.items() if members & fav_genres
        }
        if song_family and song_family in user_families:
            genre_score = 0.25
            reasons.append(
                f"genre '{song_genre}' shares family '{song_family}' with favorites (quarter points)"
            )
        else:
            genre_score = 0.0
            reasons.append(
                f"genre '{song_genre}' does not match favorites (no points)")

    # ── S2: Mood Match (weight 0.20) ─────────────────────────────────────────
    fav_mood = user_prefs.get("favorite_mood", "").lower()
    song_mood = song.get("mood", "").lower()
    if song_mood == fav_mood:
        mood_score = 1.0
        reasons.append(f"mood '{song_mood}' matches preferred mood")
    else:
        mood_score = 0.0
        reasons.append(
            f"mood '{song_mood}' does not match preferred mood '{fav_mood}'")

    # ── S3: Energy Proximity (weight 0.30) ────────────────────────────────────
    target_energy = float(user_prefs.get("target_energy", 0.5))
    energy_score = 1.0 - abs(float(song.get("energy", 0.5)) - target_energy)
    reasons.append(
        f"energy proximity {energy_score:.2f} "
        f"(song {song.get('energy'):.2f} vs target {target_energy:.2f})"
    )

    # ── S4: Acoustic Preference (weight 0.15) ─────────────────────────────────
    target_acousticness = float(user_prefs.get("target_acousticness", 0.5))
    acoustic_score = 1.0 - \
        abs(float(song.get("acousticness", 0.5)) - target_acousticness)
    reasons.append(
        f"acousticness proximity {acoustic_score:.2f} "
        f"(song {song.get('acousticness'):.2f} vs target {target_acousticness:.2f})"
    )

    # ── S5: Valence Proximity (weight 0.10) ───────────────────────────────────
    target_valence = MOOD_VALENCE_TARGET.get(fav_mood, 0.5)
    valence_score = 1.0 - abs(float(song.get("valence", 0.5)) - target_valence)
    reasons.append(
        f"valence proximity {valence_score:.2f} "
        f"(song {song.get('valence'):.2f} vs mood-mapped target {target_valence:.2f})"
    )

    # ── Weighted sum ──────────────────────────────────────────────────────────
    total = (
        genre_score * 0.25
        + mood_score * 0.20
        + energy_score * 0.30
        + acoustic_score * 0.15
        + valence_score * 0.10
    )

    return (round(total, 4), reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Ranking rules (from README):
      1. Sort descending by score
      2. Tie-break by catalog order (stable sort preserves original index)
      3. Return top-k results
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = " | ".join(reasons)
        scored.append((song, score, explanation))

    # Python's sort is stable, so equal scores retain their original catalog order
    scored.sort(key=lambda entry: entry[1], reverse=True)

    return scored[:k]

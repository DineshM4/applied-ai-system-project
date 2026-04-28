import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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


def _float_label(val: float) -> str:
    """Maps a 0.0–1.0 float to a plain-English intensity word."""
    if val >= 0.75:
        return "very high"
    if val >= 0.50:
        return "high"
    if val >= 0.25:
        return "moderate"
    return "low"


def song_to_document(song: Dict) -> str:
    """Converts a song dict into a natural language string for ChromaDB embedding."""
    return (
        f"{song['title']} by {song['artist']} is a {song['mood']} {song['genre']} track "
        f"with {_float_label(song['energy'])} energy, "
        f"{_float_label(song['acousticness'])} acousticness, "
        f"{_float_label(song['valence'])} valence, "
        f"and a tempo of {song['tempo_bpm']} BPM."
    )


def build_chroma_collection(songs: List[Dict]):
    """Loads all songs into an in-memory ChromaDB collection and returns it."""
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("songs")
    collection.add(
        ids=[str(s["id"]) for s in songs],
        documents=[song_to_document(s) for s in songs],
        metadatas=[{k: str(v) for k, v in s.items()} for s in songs],
    )
    return collection


def semantic_recommend(query: str, collection, k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Queries ChromaDB with a natural language string and returns top-k matches.

    Returns list of (song_dict, distance, document_string).
    Distance is 0–2 cosine distance — lower means closer match.
    """
    results = collection.query(query_texts=[query], n_results=k)
    output = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        song_dict = {
            "id":           int(metadata["id"]),
            "title":        metadata["title"],
            "artist":       metadata["artist"],
            "genre":        metadata["genre"],
            "mood":         metadata["mood"],
            "energy":       float(metadata["energy"]),
            "tempo_bpm":    float(metadata["tempo_bpm"]),
            "valence":      float(metadata["valence"]),
            "danceability": float(metadata["danceability"]),
            "acousticness": float(metadata["acousticness"]),
        }
        distance = results["distances"][0][i]
        document = results["documents"][0][i]
        output.append((song_dict, distance, document))
    return output


def generate_explanation(query: str, rag_results: List[Tuple]) -> str:
    """Calls Gemini to produce a DJ-style explanation for the retrieved songs.

    Raises on any API error so the caller's guardrail can catch and fall back.
    """
    from google import genai

    context = "\n".join(f"- {doc}" for _, _, doc in rag_results)
    prompt = (
        f"You are an expert DJ assistant. A user asked for: '{query}'.\n"
        f"Based on a semantic search of our music catalog, these songs were retrieved:\n"
        f"{context}\n\n"
        f"Write a short, enthusiastic playlist recommendation. "
        f"For each song write 1-2 sentences explaining why it fits the user's request. "
        f"Reference the genre, mood, and energy of each track specifically."
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return response.text


NEUTRAL_PROFILE = {
    "name": "Fallback",
    "favorite_genre": [],
    "favorite_mood": "chill",
    "target_energy": 0.5,
    "target_acousticness": 0.5,
}


def rag_recommend(query: str, collection, songs: List[Dict], k: int = 5) -> Dict:
    """Top-level RAG pipeline with guardrail fallback.

    Tries: ChromaDB semantic retrieval → Gemini explanation.
    Falls back to score_song math if the LLM call fails for any reason.

    Always returns a dict with keys: source, songs, explanation.
    """
    try:
        rag_results = semantic_recommend(query, collection, k)
        explanation = generate_explanation(query, rag_results)
        return {
            "source": "rag",
            "songs": [s for s, _, _ in rag_results],
            "explanation": explanation,
        }
    except Exception as e:
        logging.warning(
            f"[FALLBACK] LLM failed ({type(e).__name__}: {e}). Falling back to score_song math.")
        fallback = recommend_songs(NEUTRAL_PROFILE, songs, k)
        explanation = "\n".join(
            f"{s['title']} by {s['artist']}: {exp}" for s, _, exp in fallback
        )
        return {
            "source": "fallback",
            "songs": [s for s, _, _ in fallback],
            "explanation": explanation,
        }


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

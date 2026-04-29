import os
import csv
import json
import logging
import chromadb
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@dataclass
class Song:
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
    def __init__(self, songs: List[Song]):
        """Stores the song catalog used for scoring and recommendation."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top-k songs ranked by score against the given user profile."""
        user_dict = asdict(user)
        scored = [(song, score_song(user_dict, asdict(song))[0])
                  for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a pipe-delimited string of scoring reasons for a song against the user profile."""
        _, reasons = score_song(asdict(user), asdict(song))
        return " | ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """Parses a CSV file into a list of song dicts, casting numeric fields to int or float."""
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


_MOOD_PHRASES = {
    "euphoric":    "euphoric and exhilarating",
    "happy":       "uplifting and joyful",
    "energetic":   "driving and high-energy",
    "aggressive":  "aggressive and hard-hitting",
    "intense":     "intense and powerful",
    "romantic":    "warm and romantic",
    "laid-back":   "relaxed and effortless",
    "chill":       "calm and easy-going",
    "focused":     "focused and minimal",
    "moody":       "dark and atmospheric",
    "melancholic": "melancholic and reflective",
    "nostalgic":   "nostalgic and bittersweet",
    "peaceful":    "serene and tranquil",
    "relaxed":     "smooth and unhurried",
    "sad":         "sad and introspective",
    "spiritual":   "contemplative and meditative",
}


def _tempo_phrase(bpm: float) -> str:
    """Maps a BPM value to a human-readable tempo label."""
    if bpm < 70:
        return "very slow"
    if bpm < 90:
        return "slow"
    if bpm < 110:
        return "mid-tempo"
    if bpm < 130:
        return "upbeat"
    if bpm < 150:
        return "fast"
    return "very fast"


def _use_case(energy: float, acousticness: float, mood: str) -> str:
    """Derives a suggested listening context from a track's energy, acousticness, and mood."""
    if energy < 0.4 and acousticness > 0.6:
        return "ideal for studying, reading, or winding down"
    if energy < 0.4:
        return "good for quiet background listening or relaxation"
    if energy > 0.8 and acousticness < 0.3:
        return "built for workouts, high-intensity activity, or releasing energy"
    if energy > 0.7 and mood in ("euphoric", "happy", "energetic"):
        return "perfect for dancing, parties, or getting motivated"
    if mood in ("melancholic", "sad", "moody"):
        return "suits late-night reflection or emotional listening"
    if mood == "romantic":
        return "fits intimate settings or slow evenings"
    if mood in ("focused", "peaceful", "spiritual"):
        return "suited to concentration, meditation, or background ambiance"
    return "versatile for casual or mood-matching listening"


def _instrumentation(acousticness: float) -> str:
    """Maps an acousticness value to a description of the track's instrumentation mix."""
    if acousticness >= 0.85:
        return "almost entirely acoustic with natural, organic textures"
    if acousticness >= 0.60:
        return "predominantly acoustic with light electronic elements"
    if acousticness >= 0.35:
        return "a blend of acoustic and electronic instrumentation"
    if acousticness >= 0.15:
        return "primarily electronic with minimal acoustic presence"
    return "fully electronic and synthetic with no acoustic instruments"


def _valence_phrase(valence: float) -> str:
    """Maps a valence value to a phrase describing the track's emotional tone."""
    if valence >= 0.80:
        return "strongly positive and feel-good"
    if valence >= 0.60:
        return "generally bright in tone"
    if valence >= 0.45:
        return "emotionally neutral or bittersweet"
    if valence >= 0.25:
        return "leaning dark or tense"
    return "heavy and emotionally weighted"


def song_to_document(song: Dict) -> str:
    """Converts a song dict into a natural language description used as the ChromaDB embedding document."""
    mood = song["mood"]
    energy = float(song["energy"])
    acousticness = float(song["acousticness"])
    return (
        f"{song['title']} by {song['artist']} is a "
        f"{_MOOD_PHRASES.get(mood, mood)} {song['genre']} track "
        f"with a {_tempo_phrase(float(song['tempo_bpm']))} tempo of {song['tempo_bpm']} BPM. "
        f"The sound is {_instrumentation(acousticness)}. "
        f"It feels {_valence_phrase(float(song['valence']))} "
        f"and is {_use_case(energy, acousticness, mood)}."
    )


def build_chroma_collection(songs: List[Dict]):
    """Creates an in-memory ChromaDB collection, adds all song documents, and returns it."""
    client = chromadb.Client()
    collection = client.get_or_create_collection("songs")
    collection.add(
        ids=[str(s["id"]) for s in songs],
        documents=[song_to_document(s) for s in songs],
        metadatas=[{k: str(v) for k, v in s.items()} for s in songs],
    )
    return collection


def semantic_recommend(query: str, collection, k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Queries ChromaDB and returns top-k matches as (song_dict, cosine_distance, document) tuples."""
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
    """Calls Gemini to produce a DJ-style playlist explanation for the retrieved songs. Raises on API error."""
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
        model="gemini-2.5-flash-lite", contents=prompt)
    return response.text


NEUTRAL_PROFILE = {
    "name": "Fallback",
    "favorite_genre": [],
    "favorite_mood": "chill",
    "target_energy": 0.5,
    "target_acousticness": 0.5,
}


def extract_profile_from_query(query: str) -> Dict:
    """Uses Gemini to parse a free-text query into a structured preference dict with favorite_genre, favorite_mood, target_energy, and target_acousticness. Raises on API or JSON parse error."""
    prompt = (
        f"Extract music preferences from this natural language query: '{query}'\n\n"
        f"Return ONLY a valid JSON object with these exact keys:\n"
        f'  "favorite_genre": array of 1-3 genre strings\n'
        f'  "favorite_mood": single mood string\n'
        f'  "target_energy": float 0.0-1.0\n'
        f'  "target_acousticness": float 0.0-1.0\n\n'
        f"Valid genres: lofi, pop, rock, ambient, jazz, synthwave, indie pop, "
        f"hip-hop, r&b, folk, metal, country, classical, electronic, reggae, emo, world\n"
        f"Valid moods: euphoric, happy, energetic, aggressive, intense, romantic, "
        f"laid-back, chill, focused, moody, melancholic, nostalgic, peaceful, relaxed, sad, spiritual\n\n"
        f"No markdown, no explanation. Return only the JSON object."
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt)
    text = response.text.strip().removeprefix(
        "```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def rerank_candidates(candidates: List[Tuple], profile: Optional[Dict], k: int) -> List[Tuple]:
    """Re-ranks candidates by semantic score alone (profile=None) or 50/50 hybrid with score_song math, returning top-k."""
    results = []
    for song_dict, distance, document in candidates:
        semantic_score = max(0.0, 1.0 - distance / 2.0)
        if profile is not None:
            math_score, _ = score_song(profile, song_dict)
            combined = 0.5 * semantic_score + 0.5 * math_score
        else:
            combined = semantic_score
        results.append((song_dict, combined, document))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def rag_recommend(query: str, collection, songs: List[Dict], k: int = 5) -> Dict:
    """Runs the full RAG pipeline: ChromaDB retrieval, Gemini profile extraction, hybrid re-ranking, and explanation generation. Falls back to score_song math on error. Always returns a dict with keys: source, songs, explanation."""
    try:
        n_candidates = min(k * 3, len(songs))
        candidates = semantic_recommend(query, collection, n_candidates)

        try:
            soft_profile = extract_profile_from_query(query)
        except Exception as e:
            logging.warning(
                f"[PROFILE EXTRACT] Failed ({type(e).__name__}). Ranking by semantic score only.")
            soft_profile = None

        reranked = rerank_candidates(candidates, soft_profile, k)
        try:
            explanation = generate_explanation(query, reranked)
        except Exception as e:
            logging.warning(
                f"[EXPLAIN FALLBACK] Gemini explanation failed ({type(e).__name__}). Using document text.")
            explanation = "\n".join(
                f"{s['title']} by {s['artist']}: {doc}" for s, _, doc in reranked
            )
        return {
            "source": "rag",
            "songs": [s for s, _, _ in reranked],
            "explanation": explanation,
        }
    except Exception as e:
        logging.warning(
            f"[FALLBACK] RAG pipeline failed ({type(e).__name__}: {e}). Falling back to score_song math.")
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
    """Scores a song against user preferences and returns (score, reasons). Score is 0.0–1.0 weighted across genre, mood, energy, acousticness, and valence."""
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
    """Scores every song against user preferences and returns the top-k as (song, score, explanation) tuples. Ties are broken by catalog order."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = " | ".join(reasons)
        scored.append((song, score, explanation))

    # Python's sort is stable, so equal scores retain their original catalog order
    scored.sort(key=lambda entry: entry[1], reverse=True)

    return scored[:k]

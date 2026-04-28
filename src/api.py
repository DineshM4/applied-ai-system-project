from recommender import load_songs, build_chroma_collection, rag_recommend
import sys
import os
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)

app_state: dict = {}


_W = 52


def _print_startup_banner(song_count: int) -> None:
    """Prints a formatted startup banner showing song count and available endpoints."""
    bar = "═" * _W
    print(f"\n{bar}")
    print("  MUSIC RECOMMENDER API  —  Ready")
    print(f"  {'─' * (_W - 4)}")
    print(f"  Songs loaded : {song_count}")
    print(f"  Endpoints    : GET /  |  POST /recommend")
    print(f"  Docs         : http://127.0.0.1:8000/docs")
    print(f"{bar}\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads songs and builds the ChromaDB collection on startup; clears state on shutdown."""
    songs = load_songs("data/songs.csv")
    collection = build_chroma_collection(songs)
    app_state["songs"] = songs
    app_state["collection"] = collection
    logging.info(f"[STARTUP] Loaded {len(songs)} songs into ChromaDB.")
    _print_startup_banner(len(songs))
    yield
    app_state.clear()


app = FastAPI(title="Music Recommender API", lifespan=lifespan)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1,
                       description="Natural language music query")
    k: int = Field(default=5, ge=1, le=20)


class SongOut(BaseModel):
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


class RecommendResponse(BaseModel):
    source: str
    songs: List[SongOut]
    explanation: str


@app.get("/")
def health():
    """Returns server status and the count of songs currently in memory."""
    return {"status": "ok", "songs_loaded": len(app_state.get("songs", []))}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Runs the RAG pipeline for the given query and returns top-k song recommendations."""
    if not app_state.get("collection"):
        raise HTTPException(status_code=503, detail="Catalog not loaded yet")
    result = rag_recommend(
        req.query,
        app_state["collection"],
        app_state["songs"],
        k=req.k,
    )
    return RecommendResponse(**result)

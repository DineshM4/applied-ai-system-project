import sys
import os
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from recommender import load_songs, build_chroma_collection, rag_recommend

logging.basicConfig(level=logging.INFO)

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    songs = load_songs("data/songs.csv")
    collection = build_chroma_collection(songs)
    app_state["songs"] = songs
    app_state["collection"] = collection
    logging.info(f"[STARTUP] Loaded {len(songs)} songs into ChromaDB.")
    yield
    app_state.clear()


app = FastAPI(title="Music Recommender API", lifespan=lifespan)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language music query")
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
    return {"status": "ok", "songs_loaded": len(app_state.get("songs", []))}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not app_state.get("collection"):
        raise HTTPException(status_code=503, detail="Catalog not loaded yet")
    result = rag_recommend(
        req.query,
        app_state["collection"],
        app_state["songs"],
        k=req.k,
    )
    return RecommendResponse(**result)

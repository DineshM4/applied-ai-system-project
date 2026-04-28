from recommender import load_songs, build_chroma_collection, rag_recommend

DIVIDER = "─" * 52
THIN_DIVIDER = "·" * 52
THICK_DIVIDER = "═" * 52

SEMANTIC_QUERIES = [
    "I want something chill and acoustic to study to",
    "Give me aggressive high-energy rock",
    "Upbeat electronic music for a workout",
]


def print_rag_result(query: str, result: dict) -> None:
    """Prints a formatted block showing the query, ranked song list, and Gemini explanation."""
    source_label = f"[{result['source'].upper()}]"
    print(f"\n{DIVIDER}")
    print(f"  QUERY  {source_label}")
    print(f'  "{query}"')
    print(DIVIDER)

    for rank, song in enumerate(result["songs"], start=1):
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(
            f"       Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']:.2f}  |  BPM: {song['tempo_bpm']:.0f}")

    print(f"\n  {THIN_DIVIDER}")
    print("  Explanation:")
    for line in result["explanation"].strip().splitlines():
        print(f"    {line}")
    print(f"\n{DIVIDER}")


def run_semantic_demo() -> None:
    """Runs the preset demo queries then enters an interactive loop for free-text RAG queries."""
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


if __name__ == "__main__":
    run_semantic_demo()

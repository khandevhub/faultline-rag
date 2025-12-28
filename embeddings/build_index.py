import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_chunks():
    all_chunks = []

    for file in CHUNK_DIR.glob("*_chunks.json"):
        with open(file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks found to embed.")

    return all_chunks


def main():
    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    print(f"[INFO] Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_DIR / "chunks.index"))

    # Save metadata aligned by index position
    with open(INDEX_DIR / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"[OK] FAISS index built with {index.ntotal} vectors")


if __name__ == "__main__":
    main()

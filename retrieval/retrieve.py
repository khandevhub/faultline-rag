import pickle
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "chunks.index"
METADATA_PATH = INDEX_DIR / "chunks_metadata.pkl"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.55

# Hybrid weights
ALPHA = 0.7  # semantic weight
BETA = 0.3   # lexical weight

SECTION_WEIGHTS = {
    "root_cause": 1.2,
    "resolution": 1.1,
    "summary": 1.0,
    "lessons_learned": 0.9,
}


def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9\-]{3,}", text.lower()))


def lexical_score(query_tokens, chunk_tokens):
    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def retrieve(query: str):
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_tokens = tokenize(query)

    # Semantic embedding
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_vec, TOP_K)

    results = []

    for sem_score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = metadata[idx].copy()

        # Lexical overlap
        chunk_tokens = tokenize(chunk["text"])
        lex_score = lexical_score(query_tokens, chunk_tokens)

        # Hybrid score
        hybrid_score = (ALPHA * float(sem_score)) + (BETA * lex_score)

        # Section weighting
        hybrid_score *= SECTION_WEIGHTS.get(chunk["section_type"], 1.0)

        if hybrid_score < SIMILARITY_THRESHOLD:
            continue

        chunk["semantic_score"] = float(sem_score)
        chunk["lexical_score"] = float(lex_score)
        chunk["hybrid_score"] = float(hybrid_score)

        results.append(chunk)

    # Sort by hybrid score
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return results

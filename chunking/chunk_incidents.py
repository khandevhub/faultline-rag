import json
from pathlib import Path
from typing import List, Dict


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized"
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


SECTION_FIELDS = {
    "summary": "summary",
    "root_cause": "root_cause",
    "resolution": "resolution",
    "lessons_learned": "lessons_learned",
}


def chunk_incident(doc: Dict) -> List[Dict]:
    chunks = []

    for section_name, field in SECTION_FIELDS.items():
        content = doc.get(field)
        if not content:
            continue

        chunk = {
            "chunk_id": f"{doc['doc_id']}::{section_name}",
            "incident_id": doc["doc_id"],
            "section_type": section_name,
            "service": doc["service"],
            "severity": doc["severity"],
            "timestamp": doc["timestamp"],
            "text": content,
            "source_url": doc["source_url"],
            "is_synthetic": doc["is_synthetic"],
        }

        chunks.append(chunk)

    return chunks


def process_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    chunks = chunk_incident(doc)

    if not chunks:
        print(f"[WARN] No chunks created for {doc['doc_id']}")
        return

    out_path = CHUNK_DIR / f"{doc['doc_id']}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"[OK] Created {len(chunks)} chunks for {doc['doc_id']}")


def main():
    files = list(NORMALIZED_DIR.glob("*.json"))
    if not files:
        raise RuntimeError("No normalized documents found.")

    for file in files:
        process_file(file)


if __name__ == "__main__":
    main()

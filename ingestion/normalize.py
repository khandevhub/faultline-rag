import json
from pathlib import Path
from schema import IncidentDocument


# Resolve project root regardless of where script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "normalized"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Validate and normalize using strict schema
    doc = IncidentDocument(**raw)

    out_path = OUT_DIR / f"{doc.doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, default=str)

    print(f"[OK] Normalized {doc.doc_id}")


def main():
    if not RAW_DIR.exists():
        raise RuntimeError(f"Raw data directory not found: {RAW_DIR}")

    files = list(RAW_DIR.glob("*.json"))
    if not files:
        print("[WARN] No raw JSON files found.")
        return

    for file in files:
        normalize_file(file)


if __name__ == "__main__":
    main()


from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from retrieval.retrieve import retrieve

# CPU-safe ~1GB model
MODEL_NAME = "google/flan-t5-base"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


def build_prompt(query, chunks):
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"Incident {c['incident_id']} ({c['section_type']}): {c['text']}"
        )

    context = "\n".join(context_blocks)

    prompt = f"""
Answer the question using ONLY the information below.
If the information is insufficient, say so clearly.
Cite incident IDs.

Information:
{context}

Question:
{query}

Answer:
""".strip()

    return prompt


def generate_answer(query):
    chunks = retrieve(query)

    if not chunks:
        return "Insufficient historical evidence to answer this question."

    tokenizer, model = load_model()
    prompt = build_prompt(query, chunks)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("Enter query (or 'exit'):")
    while True:
        query = input("> ").strip()
        if query.lower() == "exit":
            break

        answer = generate_answer(query)
        print("\n[ANSWER]")
        print(answer)
        print()


if __name__ == "__main__":
    main()

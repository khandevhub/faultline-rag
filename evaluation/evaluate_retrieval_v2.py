import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from retrieval.retrieve import retrieve


# Evaluation dataset
EVAL_QUERIES = [
    {
        "query": "What caused the 502 errors after deployment?",
        "relevant_incidents": {"incident_001"},
        "should_answer": True,
    },
    {
        "query": "timeout misconfiguration upstream services",
        "relevant_incidents": {"incident_001"},
        "should_answer": True,
    },
    {
        "query": "API gateway returned 502 after config change",
        "relevant_incidents": {"incident_001"},
        "should_answer": True,
    },
    {
        "query": "database replication lag",
        "relevant_incidents": set(),
        "should_answer": False,
    },
    {
        "query": "kafka consumer lag high",
        "relevant_incidents": set(),
        "should_answer": False,
    },
]


TOP_K = 3


def precision_at_k(retrieved, relevant, k):
    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    retrieved_ids = {r["incident_id"] for r in top_k}

    return len(retrieved_ids & relevant) / min(k, len(retrieved_ids))


def evaluate():
    print("\n=== Retrieval Evaluation v2 ===\n")

    total_answer_queries = 0
    correct_answer_queries = 0

    total_refusal_queries = 0
    correct_refusals = 0

    false_positives = 0
    precision_scores = []

    for test in EVAL_QUERIES:
        query = test["query"]
        relevant = test["relevant_incidents"]
        should_answer = test["should_answer"]

        results = retrieve(query)

        print(f"Query: {query}")

        if should_answer:
            total_answer_queries += 1

            if not results:
                print("  ❌ FAIL — system refused but should answer")
                continue

            p_at_k = precision_at_k(results, relevant, TOP_K)
            precision_scores.append(p_at_k)

            retrieved_ids = {r["incident_id"] for r in results}

            if retrieved_ids & relevant:
                correct_answer_queries += 1
                print(f"  ✅ PASS — retrieved {retrieved_ids}")
                print(f"  Precision@{TOP_K}: {p_at_k:.2f}")
            else:
                false_positives += 1
                print(f"  ❌ FAIL — wrong incidents {retrieved_ids}")

        else:
            total_refusal_queries += 1

            if results:
                false_positives += 1
                print(f"  ❌ FAIL — should refuse but retrieved { {r['incident_id'] for r in results} }")
            else:
                correct_refusals += 1
                print("  ✅ PASS — correctly refused")

        print()

    avg_precision = (
        sum(precision_scores) / len(precision_scores)
        if precision_scores else 0.0
    )

    print("=== Summary ===")
    print(f"Answer queries correct: {correct_answer_queries}/{total_answer_queries}")
    print(f"Refusals correct: {correct_refusals}/{total_refusal_queries}")
    print(f"False positives: {false_positives}")
    print(f"Average Precision@{TOP_K}: {avg_precision:.2f}")


if __name__ == "__main__":
    evaluate()

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from retrieval.retrieve import retrieve


# Simple evaluation set
TEST_QUERIES = [
    {
        "query": "What caused the 502 errors after deployment?",
        "expected_incident": "incident_001",
        "should_answer": True
    },
    {
        "query": "timeout misconfiguration upstream services",
        "expected_incident": "incident_001",
        "should_answer": True
    },
    {
        "query": "database replication lag",
        "expected_incident": None,
        "should_answer": False
    },
    {
        "query": "how to reduce kafka consumer lag",
        "expected_incident": None,
        "should_answer": False
    }
]


def evaluate():
    print("Running retrieval evaluation...\n")

    passed = 0

    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected_incident"]
        should_answer = test["should_answer"]

        results = retrieve(query)

        if should_answer:
            if not results:
                print(f"[FAIL] '{query}' → No results returned")
                continue

            incident_ids = {r["incident_id"] for r in results}
            if expected in incident_ids:
                print(f"[PASS] '{query}' → Retrieved {incident_ids}")
                passed += 1
            else:
                print(f"[FAIL] '{query}' → Wrong incident {incident_ids}")
        else:
            if results:
                print(f"[FAIL] '{query}' → Should have refused, got results")
            else:
                print(f"[PASS] '{query}' → Correctly refused")
                passed += 1

    print(f"\nSummary: {passed}/{len(TEST_QUERIES)} tests passed")


if __name__ == "__main__":
    evaluate()

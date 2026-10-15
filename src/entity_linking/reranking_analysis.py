
from src.entity_linking.candidate_retriever import CandidateRetriever
from src.entity_linking.evaluation import entities

TEST_CASES = [
    {
        "mention": "Mercury",
        "context": "The planet completed its orbit around the Sun.",
        "gold": "PLANET_MERCURY",
    },
    {
        "mention": "Mercury",
        "context": "The sample contained toxic Hg metal.",
        "gold": "ELEMENT_MERCURY",
    },
    {
        "mention": "Python",
        "context": "The developer debugged code using machine learning libraries.",
        "gold": "LANG_PYTHON",
    },
    {
        "mention": "Python",
        "context": "The reptile constricted its prey in the forest.",
        "gold": "ANIMAL_PYTHON",
    },
]


def reciprocal_rank(candidates, gold_id):
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["entity_id"] == gold_id:
            return 1.0 / rank
    return 0.0


def main():
    retriever = CandidateRetriever()
    retriever.add_entities(entities)

    baseline_correct = 0
    reranked_correct = 0
    fixed_errors = 0
    introduced_errors = 0
    baseline_rr = []
    reranked_rr = []

    for case in TEST_CASES:
        candidates = retriever.search(
            mention=case["mention"],
            context=case["context"],
            top_k=5,
        )

        # Keep the retrieved candidate set identical.
        baseline_ids = [
            item["entity_id"] for item in candidates
        ]

        reranked = retriever.rerank_candidates(
            mention=case["mention"],
            context=case["context"],
            candidates=candidates,
            top_k=5,
        )

        reranked_ids = [
            item["entity_id"] for item in reranked
        ]

        gold = case["gold"]
        baseline_ok = bool(baseline_ids and baseline_ids[0] == gold)
        reranked_ok = bool(reranked_ids and reranked_ids[0] == gold)

        baseline_correct += int(baseline_ok)
        reranked_correct += int(reranked_ok)
        fixed_errors += int(not baseline_ok and reranked_ok)
        introduced_errors += int(baseline_ok and not reranked_ok)

        baseline_rr.append(reciprocal_rank(candidates, gold))
        reranked_rr.append(reciprocal_rank(reranked, gold))

        print("\nMention:", case["mention"])
        print("Context:", case["context"])
        print("Gold entity:", gold)
        print("Baseline top-1:", baseline_ids[0] if baseline_ids else "None")
        print("Reranked top-1:", reranked_ids[0] if reranked_ids else "None")

    n = len(TEST_CASES)

    print("\n" + "=" * 50)
    print("RERANKING COMPARISON")
    print("=" * 50)
    print(f"Examples: {n}")
    print(f"Baseline Accuracy@1: {baseline_correct / n:.3f}")
    print(f"Reranked Accuracy@1: {reranked_correct / n:.3f}")
    print(f"Baseline MRR: {sum(baseline_rr) / n:.3f}")
    print(f"Reranked MRR: {sum(reranked_rr) / n:.3f}")
    print(f"Errors fixed by reranking: {fixed_errors}")
    print(f"Errors introduced by reranking: {introduced_errors}")


if __name__ == "__main__":
    main()

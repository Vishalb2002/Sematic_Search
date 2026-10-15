
from typing import Dict, List

from src.entity_linking.candidate_retriever import CandidateRetriever


# Small manually labeled smoke-test dataset.
# Each gold_entity_id must match a canonical entity ID.
entities = [
    {
        "entity_id": "ORG_APPLE",
        "name": "Apple Inc.",
        "entity_type": "ORG",
        "description": "Technology company producing iPhones, computers, and software.",
        "aliases": ["Apple"],
    },
    {
        "entity_id": "CONCEPT_APPLE_FRUIT",
        "name": "Apple fruit",
        "entity_type": "CONCEPT",
        "description": "Edible fruit that grows on apple trees.",
        "aliases": ["Apple", "apple fruit"],
    },
    {
        "entity_id": "ORG_GOOGLE",
        "name": "Google",
        "entity_type": "ORG",
        "description": "Technology company operating a search engine and developing AI.",
        "aliases": ["Google LLC"],
    },
    {
        "entity_id": "PLANET_MERCURY",
        "name": "Mercury (planet)",
        "entity_type": "CONCEPT",
        "description": "The smallest planet and closest planet to the Sun.",
        "aliases": ["Mercury"],
    },
    {
        "entity_id": "ELEMENT_MERCURY",
        "name": "Mercury (chemical element)",
        "entity_type": "CONCEPT",
        "description": "A metallic chemical element with symbol Hg.",
        "aliases": ["Mercury", "quicksilver"],
    },
    {
        "entity_id": "LANG_PYTHON",
        "name": "Python programming language",
        "entity_type": "CONCEPT",
        "description": "A high-level programming language used in AI and data science.",
        "aliases": ["Python"],
    },
    {
        "entity_id": "ANIMAL_PYTHON",
        "name": "Python snake",
        "entity_type": "CONCEPT",
        "description": "A nonvenomous constrictor snake.",
        "aliases": ["Python"],
    },
    {
        "entity_id": "COMP_AMAZON",
        "name": "Amazon",
        "entity_type": "ORG",
        "description": "An e-commerce and cloud computing company.",
        "aliases": ["Amazon.com", "Amazon"],
    },
    {
        "entity_id": "RIVER_AMAZON",
        "name": "Amazon River",
        "entity_type": "LOCATION",
        "description": "A major river system in South America.",
        "aliases": ["Amazon"],
    },
    {
        "entity_id": "PERSON_JORDAN",
        "name": "Michael Jordan",
        "entity_type": "PERSON",
        "description": "Former professional basketball player.",
        "aliases": ["Jordan", "Michael Jordan"],
    },
    {
        "entity_id": "COUNTRY_JORDAN",
        "name": "Jordan",
        "entity_type": "LOCATION",
        "description": "A country in the Middle East.",
        "aliases": ["Jordan"],
    },
    {
        "entity_id": "LANG_JAVA",
        "name": "Java programming language",
        "entity_type": "CONCEPT",
        "description": "A class-based programming language used to build applications.",
        "aliases": ["Java"],
    },
]


TEST_DATA = [
    {
        "mention": "Apple",
        "context": "Apple announced a new iPhone.",
        "gold_entity_id": "ORG_APPLE",
    },
    {
        "mention": "apple",
        "context": "The apple tree produced sweet fruit.",
        "gold_entity_id": "CONCEPT_APPLE_FRUIT",
    },
    {
        "mention": "Mercury",
        "context": "Mercury completes an orbit around the Sun.",
        "gold_entity_id": "PLANET_MERCURY",
    },
    {
        "mention": "Mercury",
        "context": "Mercury has the chemical symbol Hg.",
        "gold_entity_id": "ELEMENT_MERCURY",
    },
    {
        "mention": "Python",
        "context": "Python supports libraries for machine learning.",
        "gold_entity_id": "LANG_PYTHON",
    },
    {
        "mention": "python",
        "context": "The python snake swallowed its prey.",
        "gold_entity_id": "ANIMAL_PYTHON",
    },
    {
        "mention": "Amazon",
        "context": "Amazon expanded its cloud computing services.",
        "gold_entity_id": "COMP_AMAZON",
    },
    {
        "mention": "Amazon",
        "context": "The Amazon flows through South America.",
        "gold_entity_id": "RIVER_AMAZON",
    },
    {
        "mention": "Jordan",
        "context": "Jordan scored points during his basketball career.",
        "gold_entity_id": "PERSON_JORDAN",
    },
    {
        "mention": "Jordan",
        "context": "Amman is the capital of Jordan.",
        "gold_entity_id": "COUNTRY_JORDAN",
    },
    {
        "mention": "Java",
        "context": "Java supports object-oriented application development.",
        "gold_entity_id": "LANG_JAVA",
    },
    {
        "mention": "Google",
        "context": "Google developed products for web search and AI.",
        "gold_entity_id": "ORG_GOOGLE",
    },
]


def reciprocal_rank(
    candidates: List[Dict],
    gold_entity_id: str,
) -> float:
    """Return the reciprocal rank of the correct entity."""
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["entity_id"] == gold_entity_id:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    candidates: List[Dict],
    gold_entity_id: str,
    k: int,
) -> float:
    """Return 1 if the correct entity occurs within the top k."""
    return float(
        any(
            candidate["entity_id"] == gold_entity_id
            for candidate in candidates[:k]
        )
    )


def evaluate_configuration(
    retriever: CandidateRetriever,
    test_data: List[Dict],
    use_reranker: bool,
    candidate_k: int = 5,
) -> Dict:
    reciprocal_ranks = []
    recall1 = []
    recall5 = []
    correct_top1 = 0
    candidate_recall = []

    method_name = (
        "SBERT + CrossEncoder"
        if use_reranker
        else "SBERT retrieval baseline"
    )

    print(f"\nDetailed results: {method_name}")
    print("-" * 70)

    for example in test_data:
        candidates = retriever.search(
            mention=example["mention"],
            context=example["context"],
            top_k=candidate_k,
        )

        gold_id = example["gold_entity_id"]

        # Measure whether retrieval found the gold entity before
        # applying reranking.
        candidate_recall.append(
            recall_at_k(candidates, gold_id, candidate_k)
        )

        if use_reranker and candidates:
            candidates = retriever.rerank_candidates(
                mention=example["mention"],
                context=example["context"],
                candidates=candidates,
                top_k=candidate_k,
            )

        ranked_ids = [
            candidate["entity_id"]
            for candidate in candidates
        ]

        gold_rank = (
            ranked_ids.index(gold_id) + 1
            if gold_id in ranked_ids
            else None
        )

        print(
            f"Mention={example['mention']!r} | "
            f"Gold={gold_id} | "
            f"Rank={gold_rank} | "
            f"Candidates={ranked_ids}"
        )

        rr = reciprocal_rank(candidates, gold_id)
        r1 = recall_at_k(candidates, gold_id, 1)
        r5 = recall_at_k(candidates, gold_id, 5)

        reciprocal_ranks.append(rr)
        recall1.append(r1)
        recall5.append(r5)
        correct_top1 += int(r1 == 1.0)

    n = len(test_data)

    return {
        "examples": n,
        "accuracy_at_1": correct_top1 / n if n else 0.0,
        "recall_at_1": sum(recall1) / n if n else 0.0,
        "recall_at_5": sum(recall5) / n if n else 0.0,
        "mrr": (
            sum(reciprocal_ranks) / n if n else 0.0
        ),
        "candidate_recall_at_5": (
            sum(candidate_recall) / n if n else 0.0
        ),
    }


def print_metrics(name: str, metrics: Dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Examples:             {metrics['examples']}")
    print(f"Accuracy@1:           {metrics['accuracy_at_1']:.3f}")
    print(f"Recall@1:             {metrics['recall_at_1']:.3f}")
    print(f"Recall@5:             {metrics['recall_at_5']:.3f}")
    print(f"MRR:                  {metrics['mrr']:.3f}")
    print(
        "Candidate Recall@5:   "
        f"{metrics['candidate_recall_at_5']:.3f}"
    )


def main():
    retriever = CandidateRetriever()

    # IMPORTANT: Use the complete catalogue defined above.
    # Do not define a second, smaller `entities` list here.
    retriever.add_entities(entities)

    print(
        "\nEntity catalogue count:",
        retriever.collection.count(),
    )

    expected_ids = {entity["entity_id"] for entity in entities}

    # Fail early if the catalogue has missing records.
    stored = retriever.collection.get(include=["metadatas"])
    stored_ids = {
        metadata["entity_id"]
        for metadata in (stored.get("metadatas") or [])
    }

    missing_ids = expected_ids - stored_ids

    if missing_ids:
        raise RuntimeError(
            f"Entity catalogue is missing IDs: {sorted(missing_ids)}"
        )

    baseline = evaluate_configuration(
        retriever,
        TEST_DATA,
        use_reranker=False,
        candidate_k=5,
    )

    reranked = evaluate_configuration(
        retriever,
        TEST_DATA,
        use_reranker=True,
        candidate_k=5,
    )

    print("\nENTITY LINKING EVALUATION")
    print("=" * 45)

    print_metrics("SBERT retrieval baseline", baseline)
    print_metrics("SBERT + CrossEncoder", reranked)

    print("\nAbsolute metric changes")
    print("-" * 45)

    for metric in (
        "accuracy_at_1",
        "recall_at_1",
        "recall_at_5",
        "mrr",
    ):
        change = reranked[metric] - baseline[metric]
        print(f"{metric}: {change:+.3f}")

    print(
        "\nNote: This is a small manually constructed smoke test, "
        "not a research benchmark."
    )


if __name__ == "__main__":
    main()

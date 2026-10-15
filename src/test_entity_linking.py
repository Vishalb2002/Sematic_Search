
from src.entity_linking.candidate_retriever import CandidateRetriever


def main():
    retriever = CandidateRetriever()

    entities = [
        {
            "entity_id": "ORG_APPLE",
            "name": "Apple Inc.",
            "entity_type": "ORG",
            "description": (
                "Technology company that develops smartphones, "
                "computers, and software."
            ),
            "aliases": ["Apple"],
        },
        {
            "entity_id": "CONCEPT_APPLE_FRUIT",
            "name": "Apple fruit",
            "entity_type": "CONCEPT",
            "description": "An edible fruit grown on apple trees.",
            "aliases": ["Apple", "apple fruit"],
        },
        {
            "entity_id": "ORG_GOOGLE",
            "name": "Google",
            "entity_type": "ORG",
            "description": (
                "Technology company known for search engines, "
                "software, and AI research."
            ),
            "aliases": ["Google LLC"],
        },
    ]

    # Upsert makes repeated test runs safe.
    retriever.add_entities(entities)

    tests = [
        {
            "mention": "Apple",
            "context": "Apple released a new smartphone.",
        },
        {
            "mention": "Apple",
            "context": "The apple was harvested from the tree.",
        },
    ]

    for test in tests:
        mention = test["mention"]
        context = test["context"]

        print("\n" + "=" * 60)
        print("Mention:", mention)
        print("Context:", context)

        candidates = retriever.search(
            mention=mention,
            context=context,
            top_k=3,
        )

        print("\nBefore reranking:")

        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"{rank}. {candidate['name']} "
                f"| Distance: {candidate['distance']:.4f}"
            )

        reranked = retriever.rerank_candidates(
            mention=mention,
            context=context,
            candidates=candidates,
            top_k=3,
        )

        print("\nAfter reranking:")

        for rank, candidate in enumerate(reranked, start=1):
            print(
                f"{rank}. {candidate['name']} "
                f"| Score: {candidate['rerank_score']:.4f}"
            )


if __name__ == "__main__":
    main()

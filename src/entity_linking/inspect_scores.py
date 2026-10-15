
from src.entity_linking.candidate_retriever import CandidateRetriever
from src.entity_linking.evaluation import entities

TEST_CASES = [
    {
        "mention": "Mercury",
        "context": (
            "Astronomers observed Mercury orbiting the Sun "
            "using a telescope."
        ),
    },
    {
        "mention": "Mercury",
        "context": (
            "The laboratory measured mercury concentration "
            "in a contaminated water sample."
        ),
    },
    {
        "mention": "Python",
        "context": (
            "The programmer debugged Python code and installed "
            "machine learning libraries."
        ),
    },
    {
        "mention": "Python",
        "context": (
            "The snake wrapped around a tree branch "
            "before capturing its prey."
        ),
    },
]


def main():
    retriever = CandidateRetriever()
    retriever.add_entities(entities)

    for example in TEST_CASES:
        mention = example["mention"]
        context = example["context"]

        candidates = retriever.search(
            mention=mention,
            context=context,
            top_k=5,
        )

        reranked = retriever.rerank_candidates(
            mention=mention,
            context=context,
            candidates=candidates,
            top_k=5,
        )

        print("\n" + "=" * 65)
        print("Mention:", mention)
        print("Context:", context)

        print("\nSBERT retrieval order:")
        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"{rank}. {candidate['name']} "
                f"| ID={candidate['entity_id']} "
                f"| Distance={candidate['distance']:.4f}"
            )

        print("\nCrossEncoder reranking order:")
        for rank, candidate in enumerate(reranked, start=1):
            print(
                f"{rank}. {candidate['name']} "
                f"| ID={candidate['entity_id']} "
                f"| Score={candidate['rerank_score']:.4f}"
            )


if __name__ == "__main__":
    main()

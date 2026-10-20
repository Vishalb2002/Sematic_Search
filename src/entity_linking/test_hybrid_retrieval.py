
from src.entity_linking.hybrid_candidate_retriever import (
    HybridCandidateRetriever,
)


def main():
    retriever = HybridCandidateRetriever()

    examples = [
        ("LEICESTERSHIRE", "Leicestershire cricket context", "ORG"),
        ("Nasser Hussain", "Nasser Hussain scored runs", "PER"),
        ("London", "The event took place in London", "LOC"),
        ("British", "British government statement", "LOC"),
    ]

    for mention, context, entity_type in examples:
        print("\n" + "=" * 60)
        print("Mention:", mention)
        print("Expected type:", entity_type)

        results = retriever.search(
            mention=mention,
            context=context,
            top_k=10,
            entity_type=entity_type,
        )

        for rank, candidate in enumerate(results, start=1):
            print(
                rank,
                candidate.get("name"),
                candidate.get("entity_type"),
                "lexical =", candidate.get("lexical_match"),
                "semantic_rank =", candidate.get("semantic_rank"),
            )


if __name__ == "__main__":
    main()

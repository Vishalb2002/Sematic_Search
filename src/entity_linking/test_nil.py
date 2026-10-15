
from src.entity_linking.candidate_retriever import CandidateRetriever


def main():
    retriever = CandidateRetriever()

    examples = [
        {
            "mention": "Apple",
            "context": "Apple released a new iPhone.",
        },
        {
            "mention": "Zorvax",
            "context": "Zorvax is a fictional company absent from our catalogue.",
        },
    ]

    for example in examples:
        result = retriever.predict_entity(
            mention=example["mention"],
            context=example["context"],
            top_k=5,
            min_score=0.0,
        )

        print("\nMention:", result["mention"])
        print("Decision:", result["decision"])
        print("Entity:", result["entity_name"])
        print("Score:", result["score"])


if __name__ == "__main__":
    main()

from src.entity_linking.llm_entity_linker import LLMEntityLinker


def main():
    linker = LLMEntityLinker(model_name="llama3.2:3b")

    mention = "Apple"
    context = (
        "Apple announced a new iPhone and reported its quarterly "
        "revenue after strong smartphone sales."
    )

    candidates = [
        {
            "entity_id": "Q312",
            "name": "Apple Inc.",
            "entity_type": "ORG",
            "description": "Technology company that produces the iPhone.",
            "aliases": "Apple | Apple Computer",
        },
        {
            "entity_id": "Q89",
            "name": "Apple",
            "entity_type": "FOOD",
            "description": "An edible fruit.",
            "aliases": "Apple fruit",
        },
    ]

    result = linker.link(
        mention=mention,
        context=context,
        candidates=candidates,
    )

    print("\nMention:", mention)
    print("Context:", context)
    print("Selected entity ID:", result["entity_id"])
    print("Reason:", result["reason"])
    print("Valid selection:", result["valid_selection"])
    print("Raw LLM response:", result["raw_response"])


if __name__ == "__main__":
    main()

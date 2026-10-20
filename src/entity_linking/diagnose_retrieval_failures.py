
import json
from collections import Counter
from pathlib import Path

from src.entity_linking.candidate_retriever import CandidateRetriever


INPUT = Path(
    "outputs/entity_linking_validation/validation_predictions.json"
)


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    retriever = CandidateRetriever()

    # Get all entity IDs currently stored in the catalogue.
    stored = retriever.collection.get()
    catalogue_ids = {str(entity_id) for entity_id in stored["ids"]}

    categories = Counter()
    entity_types = Counter()
    examples_by_category = {
        "missing_from_catalogue": [],
        "not_retrieved_top10": [],
        "retrieved_but_not_top1": [],
        "correct_top1": [],
    }

    for example in examples:
        gold_id = str(example["gold_id"])

        # The original SBERT candidate ranking, before reranking.
        retrieved = example.get("retrieved", [])
        retrieved_ids = [
            str(candidate["entity_id"])
            for candidate in retrieved
        ]

        # Try to obtain the mention's entity type if available.
        entity_type = example.get("entity_type", "UNKNOWN")

        if gold_id not in catalogue_ids:
            category = "missing_from_catalogue"

        elif gold_id not in retrieved_ids:
            category = "not_retrieved_top10"

        elif retrieved_ids and retrieved_ids[0] == gold_id:
            category = "correct_top1"

        else:
            category = "retrieved_but_not_top1"

        categories[category] += 1
        entity_types[(entity_type, category)] += 1

        if len(examples_by_category[category]) < 10:
            examples_by_category[category].append({
                "mention": example.get("mention"),
                "gold_id": gold_id,
                "entity_type": entity_type,
                "category": category,
                "sbert_top10": retrieved_ids,
            })

    total = len(examples)

    print("\n" + "=" * 65)
    print("VALIDATION RETRIEVAL FAILURE DIAGNOSTICS")
    print("=" * 65)
    print("Validation examples:", total)
    print("Catalogue entities:", len(catalogue_ids))

    category_names = {
        "missing_from_catalogue": "1. Gold entity missing from catalogue",
        "not_retrieved_top10": "2. In catalogue, but absent from top 10",
        "retrieved_but_not_top1": "3. In top 10, but not ranked first",
        "correct_top1": "4. Correctly ranked first by SBERT",
    }

    for key, label in category_names.items():
        count = categories[key]
        percentage = 100 * count / total if total else 0

        print(f"\n{label}")
        print(f"  Count: {count}")
        print(f"  Percentage: {percentage:.2f}%")

    print("\n" + "=" * 65)
    print("EXAMPLES")
    print("=" * 65)

    for key, label in category_names.items():
        print(f"\n{label}")

        for example in examples_by_category[key]:
            print(json.dumps(example, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("COUNTS BY ENTITY TYPE")
    print("=" * 65)

    for (entity_type, category), count in sorted(
        entity_types.items()
    ):
        print(f"{entity_type:10s} | {category:25s} | {count}")

    output = Path(
        "outputs/entity_linking_validation/"
        "retrieval_failure_diagnostics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_examples": total,
        "catalogue_size": len(catalogue_ids),
        "counts": dict(categories),
        "percentages": {
            key: (
                100 * categories[key] / total
                if total else 0
            )
            for key in category_names
        },
        "examples": examples_by_category,
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved diagnostics to:", output)


if __name__ == "__main__":
    main()

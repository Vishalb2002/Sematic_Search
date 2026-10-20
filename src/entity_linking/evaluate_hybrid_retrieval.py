
import json
from pathlib import Path

from datasets import load_dataset

from src.entity_linking.hybrid_candidate_retriever import (
    HybridCandidateRetriever,
)


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
OUTPUT_DIR = Path("outputs/entity_linking_validation")


def evaluate(rankings, gold_ids):
    total = len(gold_ids)
    results = {}

    for k in (1, 5, 10):
        hits = sum(
            gold in ranking[:k]
            for ranking, gold in zip(rankings, gold_ids)
        )
        results[f"recall@{k}"] = hits / total if total else 0.0

    reciprocal_ranks = []
    top1 = 0

    for ranking, gold in zip(rankings, gold_ids):
        if ranking and ranking[0] == gold:
            top1 += 1

        try:
            reciprocal_ranks.append(
                1.0 / (ranking.index(gold) + 1)
            )
        except ValueError:
            reciprocal_ranks.append(0.0)

    results["mrr"] = (
        sum(reciprocal_ranks) / total if total else 0.0
    )
    results["top1_accuracy"] = (
        top1 / total if total else 0.0
    )

    return results


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    validation = dataset["validation"]

    print("Initializing hybrid retriever...")
    retriever = HybridCandidateRetriever()

    gold_ids = []
    hybrid_rankings = []
    baseline_rankings = []
    missing_from_catalogue = 0
    prediction_details = []

    catalogue_data = retriever.retriever.collection.get()
    catalogue_ids = {
        str(entity_id) for entity_id in catalogue_data["ids"]
    }

    total_mentions = 0

    for document in validation:
        text = document["text"]

        for entity in document["entities"]:
            qid = entity.get("qid")
            title = entity.get("title")

            # Keep the same linked-mention evaluation protocol.
            if qid is None or title is None:
                continue

            gold_id = str(qid)
            start = entity["start"]
            end = entity["end"]
            mention = text[start:end]

            # Context is deliberately kept simple and reproducible.
            context = text[
                max(0, start - 150):min(len(text), end + 150)
            ]

            entity_type = entity.get("type")

            # CandidateRetriever uses the existing entity type labels.
            # If the dataset does not provide a compatible type,
            # omit the type filter.
            type_filter = (
                entity_type
                if entity_type in {"PER", "ORG", "LOC", "MISC"}
                else None
            )

            hybrid_candidates = retriever.search(
                mention=mention,
                context=context,
                top_k=10,
                entity_type=type_filter,
                semantic_top_k=30,
            )

            hybrid_ids = [
                str(candidate["entity_id"])
                for candidate in hybrid_candidates
            ]

            # Reproduce the semantic baseline with the same query.
            semantic_candidates = retriever.retriever.search(
                mention=mention,
                context=context,
                top_k=10,
                entity_type=type_filter,
            )

            semantic_ids = [
                str(candidate["entity_id"])
                for candidate in semantic_candidates
            ]

            gold_ids.append(gold_id)
            hybrid_rankings.append(hybrid_ids)
            baseline_rankings.append(semantic_ids)
            total_mentions += 1

            if gold_id not in catalogue_ids:
                missing_from_catalogue += 1

            prediction_details.append({
                "mention": mention,
                "context": context,
                "gold_id": gold_id,
                "gold_title": title,
                "entity_type": type_filter,
                "in_catalogue": gold_id in catalogue_ids,
                "baseline_ids": semantic_ids,
                "hybrid_ids": hybrid_ids,
            })

    print("\nValidation mentions:", total_mentions)
    print(
        "Gold entities missing from catalogue:",
        missing_from_catalogue,
    )
    print(
        "Catalogue coverage:",
        round(
            1 - missing_from_catalogue / total_mentions, 4
        ) if total_mentions else 0.0,
    )

    print("\nSBERT baseline:")
    print(json.dumps(
        evaluate(baseline_rankings, gold_ids), indent=2
    ))

    print("\nHybrid lexical + SBERT:")
    print(json.dumps(
        evaluate(hybrid_rankings, gold_ids), indent=2
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "hybrid_retrieval_validation.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "total_mentions": total_mentions,
            "missing_from_catalogue": missing_from_catalogue,
            "sbert_metrics": evaluate(baseline_rankings, gold_ids),
            "hybrid_metrics": evaluate(hybrid_rankings, gold_ids),
            "predictions": prediction_details,
        }, f, indent=2, ensure_ascii=False)

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()


import random

from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
SAMPLE_SIZE = 500
RANDOM_SEED = 42
TOP_K = 10


def main():
    dataset = load_dataset(DATASET_NAME)
    retriever = CandidateRetriever()

    catalogue_result = retriever.collection.get()
    catalogue_ids = set(catalogue_result["ids"])

    if not catalogue_ids:
        raise RuntimeError(
            "Catalogue is empty. Build the training catalogue first."
        )

    # Collect linked mentions from the entire test split.
    examples = []

    for document in dataset["test"]:
        text = document["text"]

        for entity in document["entities"]:
            qid = entity.get("qid")
            title = entity.get("title")

            if qid is None or title is None:
                continue

            start = entity["start"]
            end = entity["end"]

            examples.append({
                "mention": text[start:end],
                "context": text[
                    max(0, start - 150):
                    min(len(text), end + 150)
                ],
                "gold_id": str(qid),
            })

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(
        examples,
        min(SAMPLE_SIZE, len(examples)),
    )

    print("Catalogue size:", len(catalogue_ids))
    print("Linked test mentions available:", len(examples))
    print("Random sample size:", len(sample))
    print("Random seed:", RANDOM_SEED)

    covered = 0
    candidate_hits = {1: 0, 5: 0, 10: 0}
    conditional_hits = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    conditional_rr = []

    for index, example in enumerate(sample, start=1):
        gold_id = example["gold_id"]
        is_covered = gold_id in catalogue_ids

        if is_covered:
            covered += 1

        candidates = retriever.search(
            mention=example["mention"],
            context=example["context"],
            top_k=TOP_K,
        )

        candidate_ids = [
            candidate["entity_id"] for candidate in candidates
        ]

        rank = (
            candidate_ids.index(gold_id) + 1
            if gold_id in candidate_ids
            else None
        )

        # End-to-end metrics: missing catalogue entities count as misses.
        for k in candidate_hits:
            if gold_id in candidate_ids[:k]:
                candidate_hits[k] += 1

        reciprocal_ranks.append(
            1.0 / rank if rank is not None else 0.0
        )

        # Conditional metrics: only entities present in the catalogue.
        if is_covered:
            for k in conditional_hits:
                if gold_id in candidate_ids[:k]:
                    conditional_hits[k] += 1

            conditional_rr.append(
                1.0 / rank if rank is not None else 0.0
            )

        if index <= 10:
            print(
                f"{index}. {example['mention']!r} "
                f"| Gold={gold_id} "
                f"| Covered={is_covered} "
                f"| Rank={rank}"
            )

        if index % 50 == 0:
            print(f"Processed {index}/{len(sample)}")

    n = len(sample)

    print("\n" + "=" * 60)
    print("RANDOM TEST-SAMPLE RESULTS — SBERT")
    print("=" * 60)
    print(f"Sample size: {n}")
    print(f"Catalogue-covered mentions: {covered}")
    print(f"Catalogue coverage: {covered / n:.4f}")

    print("\nEnd-to-end metrics (all sampled mentions)")
    for k in (1, 5, 10):
        print(f"Recall@{k}: {candidate_hits[k] / n:.4f}")

    print(f"End-to-end MRR: {sum(reciprocal_ranks) / n:.4f}")

    print("\nConditional metrics (catalogue-covered mentions)")
    if covered:
        for k in (1, 5, 10):
            print(
                f"Recall@{k}: "
                f"{conditional_hits[k] / covered:.4f}"
            )

        print(
            f"Conditional MRR: "
            f"{sum(conditional_rr) / covered:.4f}"
        )
    else:
        print("No sampled gold entities were present in the catalogue.")


if __name__ == "__main__":
    main()

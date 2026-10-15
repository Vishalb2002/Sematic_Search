
from collections import defaultdict

from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
MAX_EXAMPLES = 100


def main():
    dataset = load_dataset(DATASET_NAME)
    test_split = dataset["test"]

    retriever = CandidateRetriever()

    # The catalogue must have been built from training data.
    catalogue_size = retriever.collection.count()
    if catalogue_size == 0:
        raise RuntimeError(
            "Entity catalogue is empty. Run "
            "python -m src.entity_linking.build_dataset_catalogue first."
        )

    print("Training catalogue size:", catalogue_size)

    # Only use linked mentions with valid entity IDs.
    examples = []

    for document in test_split:
        text = document["text"]

        for entity in document["entities"]:
            qid = entity.get("qid")
            title = entity.get("title")

            if qid is None or title is None:
                continue

            start = entity["start"]
            end = entity["end"]
            mention = text[start:end]

            # Context includes the surrounding sentence/document text.
            context = text[
                max(0, start - 150):
                min(len(text), end + 150)
            ]

            examples.append({
                "mention": mention,
                "context": context,
                "gold_entity_id": str(qid),
                "title": title,
            })

            if len(examples) >= MAX_EXAMPLES:
                break

        if len(examples) >= MAX_EXAMPLES:
            break

    if not examples:
        raise RuntimeError("No linked test mentions found.")

    hits = defaultdict(int)
    missing_from_catalogue = 0
    ranks = []
    evaluated = 0

    print(f"\nEvaluating {len(examples)} linked test mentions...")

    for index, example in enumerate(examples, start=1):
        candidates = retriever.search(
            mention=example["mention"],
            context=example["context"],
            top_k=10,
        )

        gold_id = example["gold_entity_id"]
        candidate_ids = [
            candidate["entity_id"] for candidate in candidates
        ]

        # Some test entities may not exist in the training catalogue.
        if gold_id not in candidate_ids and not retriever.collection.get(
            ids=[gold_id]
        )["ids"]:
            missing_from_catalogue += 1
            continue

        evaluated += 1

        rank = (
            candidate_ids.index(gold_id) + 1
            if gold_id in candidate_ids
            else None
        )

        if rank is not None:
            ranks.append(1.0 / rank)

        for k in (1, 5, 10):
            if gold_id in candidate_ids[:k]:
                hits[k] += 1

        if index <= 10:
            print(
                f"{index}. Mention={example['mention']!r} "
                f"| Gold={gold_id} "
                f"| Rank={rank}"
            )

        if index % 20 == 0:
            print(f"Processed {index}/{len(examples)} mentions")

    print("\n" + "=" * 55)
    print("REAL TEST-SPLIT EVALUATION — SBERT RETRIEVAL")
    print("=" * 55)
    print(f"Requested examples: {len(examples)}")
    print(f"Evaluated examples: {evaluated}")
    print(f"Gold entities absent from catalogue: {missing_from_catalogue}")

    if evaluated:
        for k in (1, 5, 10):
            print(
                f"Recall@{k}: "
                f"{hits[k] / evaluated:.4f}"
            )

        print(f"MRR: {sum(ranks) / evaluated:.4f}")


if __name__ == "__main__":
    main()

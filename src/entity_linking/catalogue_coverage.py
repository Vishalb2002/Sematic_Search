
from collections import Counter

from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"


def main():
    dataset = load_dataset(DATASET_NAME)
    retriever = CandidateRetriever()

    # Read IDs from the training-built catalogue.
    stored = retriever.collection.get(
        include=["metadatas"]
    )

    catalogue_ids = set(stored["ids"])

    print("Catalogue entities:", len(catalogue_ids))

    for split_name in ("train", "validation", "test"):
        split = dataset[split_name]

        total_linked = 0
        covered_mentions = 0
        linked_ids = set()
        covered_ids = set()
        missing_ids = Counter()
        missing_examples = []

        for document in split:
            text = document["text"]

            for entity in document["entities"]:
                qid = entity.get("qid")
                title = entity.get("title")

                if qid is None or title is None:
                    continue

                entity_id = str(qid)
                total_linked += 1
                linked_ids.add(entity_id)

                if entity_id in catalogue_ids:
                    covered_mentions += 1
                    covered_ids.add(entity_id)
                else:
                    missing_ids[entity_id] += 1

                    if len(missing_examples) < 10:
                        start = entity["start"]
                        end = entity["end"]

                        missing_examples.append({
                            "mention": text[start:end],
                            "gold_id": entity_id,
                            "title": title,
                        })

        print(f"\n{'=' * 55}")
        print("Split:", split_name)
        print("Linked mentions:", total_linked)
        print("Unique linked entity IDs:", len(linked_ids))
        print("Mentions covered by catalogue:", covered_mentions)
        print("Mentions missing from catalogue:", total_linked - covered_mentions)

        mention_coverage = (
            covered_mentions / total_linked
            if total_linked else 0.0
        )
        unique_coverage = (
            len(covered_ids) / len(linked_ids)
            if linked_ids else 0.0
        )

        print(f"Mention coverage: {mention_coverage:.4f}")
        print(f"Unique entity coverage: {unique_coverage:.4f}")

        if split_name != "train":
            print("\nExamples missing from catalogue:")
            for example in missing_examples:
                print(example)


if __name__ == "__main__":
    main()

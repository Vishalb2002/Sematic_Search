
from collections import Counter

from datasets import load_dataset


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"


def main():
    dataset = load_dataset(DATASET_NAME)

    print("DATASET SCHEMA")
    print("=" * 60)
    print(dataset["train"].features)

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]

        entity_count = 0
        linked_count = 0
        missing_id_count = 0
        type_counts = Counter()
        unique_ids = set()
        missing_examples = []

        for document in split:
            text = document["text"]

            for entity in document["entities"]:
                entity_count += 1

                start = entity["start"]
                end = entity["end"]
                mention = text[start:end]

                entity_type = entity.get("tag")
                entity_id = entity.get("qid")
                title = entity.get("title")

                type_counts[entity_type] += 1

                if entity_id is not None:
                    linked_count += 1
                    unique_ids.add(entity_id)
                else:
                    missing_id_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append({
                            "mention": mention,
                            "type": entity_type,
                            "title": title,
                            "context": text[
                                max(0, start - 60):
                                min(len(text), end + 60)
                            ],
                        })

        print(f"\n{'=' * 60}")
        print(f"SPLIT: {split_name}")
        print(f"Documents: {len(split)}")
        print(f"Entity mentions: {entity_count}")
        print(f"Linked mentions: {linked_count}")
        print(f"Mentions without entity ID: {missing_id_count}")
        print(f"Unique linked entity IDs: {len(unique_ids)}")
        print(f"Entity types: {dict(type_counts)}")

        if missing_examples:
            print("\nExamples without entity IDs:")
            for example in missing_examples:
                print(example)


if __name__ == "__main__":
    main()

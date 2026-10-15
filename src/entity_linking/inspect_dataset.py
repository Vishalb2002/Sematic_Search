
from datasets import load_dataset


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"


def main():
    print("Loading AIDA-CoNLL-YAGO-Wikidata...")

    dataset = load_dataset(DATASET_NAME)

    print("\nAvailable splits:")
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} documents")

    print("\nFirst training document:")
    example = dataset["train"][0]

    print("Document ID:", example["document_id"])
    print("Text:", example["text"][:500])

    print("\nFirst annotated entities:")
    for entity in example["entities"][:10]:
        start = entity["start"]
        end = entity["end"]

        print({
            "mention": example["text"][start:end],
            "type": entity.get("tag"),
            "entity_id": entity.get("qid"),
            "title": entity.get("title"),
        })


if __name__ == "__main__":
    main()


from collections import defaultdict

from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"


def main():
    print("Loading AIDA-CoNLL dataset...")
    dataset = load_dataset(DATASET_NAME)

    # Build the catalogue using the training split only.
    grouped = defaultdict(
        lambda: {
            "names": set(),
            "aliases": set(),
            "types": set(),
            "contexts": [],
        }
    )

    for document in dataset["train"]:
        text = document["text"]

        for annotation in document["entities"]:
            qid = annotation.get("qid")
            title = annotation.get("title")

            # Exclude annotations without a linked entity ID.
            if qid is None or title is None:
                continue

            entity_id = str(qid)
            start = annotation["start"]
            end = annotation["end"]
            mention = text[start:end]
            entity_type = annotation.get("tag")

            record = grouped[entity_id]
            record["names"].add(title.replace("_", " "))
            record["aliases"].add(mention)

            if entity_type:
                record["types"].add(entity_type)

            # Keep a few short training contexts per entity.
            context = text[
                max(0, start - 100):
                min(len(text), end + 100)
            ].replace("\n", " ")

            if (
                context not in record["contexts"]
                and len(record["contexts"]) < 3
            ):
                record["contexts"].append(context)

    catalogue = []

    for entity_id, record in grouped.items():
        # Use a deterministic canonical name.
        name = sorted(record["names"])[0]

        entity_type = (
            sorted(record["types"])[0]
            if record["types"]
            else "UNKNOWN"
        )

        descriptions = record["contexts"]

        catalogue.append({
            "entity_id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "description": (
                f"Entity: {name}. "
                f"Type: {entity_type}. "
                f"Training contexts: "
                + " | ".join(descriptions)
            ),
            "aliases": sorted(record["aliases"]),
        })

    print(f"Unique training entities: {len(catalogue)}")

    # Replace the old small demonstration catalogue to ensure
    # that no stale test entities remain in this collection.
    retriever = CandidateRetriever()
    collection = retriever.collection

    existing_ids = collection.get(
        limit=collection.count()
    )["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)

    print("Adding training entities to ChromaDB...")
    retriever.add_entities(catalogue)

    print(
        "Catalogue size after indexing:",
        collection.count(),
    )

    print("\nSample catalogue entries:")
    for entity in catalogue[:5]:
        print({
            "entity_id": entity["entity_id"],
            "name": entity["name"],
            "entity_type": entity["entity_type"],
            "aliases": entity["aliases"][:5],
        })

    print("\nTraining-only catalogue built successfully.")


if __name__ == "__main__":
    main()

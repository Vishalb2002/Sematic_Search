
from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
TOP_K = 10


def reciprocal_rank(candidates, gold_id):
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["entity_id"] == gold_id:
            return 1.0 / rank
    return 0.0


def main():
    dataset = load_dataset(DATASET_NAME)
    retriever = CandidateRetriever()

    catalogue_ids = set(retriever.collection.get()["ids"])

    if not catalogue_ids:
        raise RuntimeError("Entity catalogue is empty.")

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

    methods = {
        name: {
            "hits": {1: 0, 5: 0, 10: 0},
            "rr": 0.0,
            "covered_hits": {1: 0, 5: 0, 10: 0},
            "covered_rr": 0.0,
        }
        for name in ("SBERT", "SBERT + CrossEncoder")
    }

    covered_count = 0
    fixed = 0
    introduced = 0
    total = len(examples)

    print("Test mentions:", total)
    print("Catalogue size:", len(catalogue_ids))
    print("Starting full test evaluation...")

    for index, example in enumerate(examples, start=1):
        gold_id = example["gold_id"]
        covered = gold_id in catalogue_ids
        covered_count += int(covered)

        candidates = retriever.search(
            mention=example["mention"],
            context=example["context"],
            top_k=TOP_K,
        )

        # Both systems use the same candidate set.
        reranked = (
            retriever.rerank_candidates(
                mention=example["mention"],
                context=example["context"],
                candidates=candidates,
                top_k=TOP_K,
            )
            if candidates else []
        )

        rankings = {
            "SBERT": candidates,
            "SBERT + CrossEncoder": reranked,
        }

        baseline_correct = bool(
            candidates and candidates[0]["entity_id"] == gold_id
        )
        reranked_correct = bool(
            reranked and reranked[0]["entity_id"] == gold_id
        )

        fixed += int(not baseline_correct and reranked_correct)
        introduced += int(baseline_correct and not reranked_correct)

        for name, ranked in rankings.items():
            ids = [item["entity_id"] for item in ranked]
            metrics = methods[name]

            for k in (1, 5, 10):
                hit = int(gold_id in ids[:k])
                metrics["hits"][k] += hit

                if covered:
                    metrics["covered_hits"][k] += hit

            rr = reciprocal_rank(ranked, gold_id)
            metrics["rr"] += rr

            if covered:
                metrics["covered_rr"] += rr

        if index % 100 == 0 or index == total:
            print(f"Processed {index}/{total}")

    print("\n" + "=" * 65)
    print("FULL AIDA-CoNLL TEST EVALUATION")
    print("=" * 65)
    print("Linked mentions:", total)
    print(f"Catalogue coverage: {covered_count / total:.4f}")

    for name, metrics in methods.items():
        print(f"\n{name} — end-to-end")
        for k in (1, 5, 10):
            print(f"Recall@{k}: {metrics['hits'][k] / total:.4f}")
        print(f"MRR: {metrics['rr'] / total:.4f}")

        print(f"\n{name} — catalogue-covered mentions")
        if covered_count:
            for k in (1, 5, 10):
                print(
                    f"Recall@{k}: "
                    f"{metrics['covered_hits'][k] / covered_count:.4f}"
                )
            print(
                f"MRR: {metrics['covered_rr'] / covered_count:.4f}"
            )

    baseline = methods["SBERT"]
    reranked = methods["SBERT + CrossEncoder"]

    print("\nAbsolute end-to-end changes")
    for k in (1, 5, 10):
        delta = (
            reranked["hits"][k] - baseline["hits"][k]
        ) / total
        print(f"Recall@{k}: {delta:+.4f}")

    print(f"MRR: {(reranked['rr'] - baseline['rr']) / total:+.4f}")
    print("Top-1 errors fixed:", fixed)
    print("Top-1 errors introduced:", introduced)


if __name__ == "__main__":
    main()


import random

from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
SAMPLE_SIZE = 500
RANDOM_SEED = 42
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

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(
        examples,
        min(SAMPLE_SIZE, len(examples)),
    )

    # Metric counters include all sampled examples.
    methods = {
        "SBERT": {
            "hits": {1: 0, 5: 0, 10: 0},
            "rr": 0.0,
        },
        "SBERT + CrossEncoder": {
            "hits": {1: 0, 5: 0, 10: 0},
            "rr": 0.0,
        },
    }

    covered_count = 0
    fixed = 0
    broken = 0

    print("Sample size:", len(sample))
    print("Catalogue size:", len(catalogue_ids))
    print("Random seed:", RANDOM_SEED)
    print("\nRunning paired comparison...")

    for index, example in enumerate(sample, start=1):
        mention = example["mention"]
        context = example["context"]
        gold_id = example["gold_id"]

        if gold_id in catalogue_ids:
            covered_count += 1

        # Retrieve candidates only once. Both methods use this set.
        candidates = retriever.search(
            mention=mention,
            context=context,
            top_k=TOP_K,
        )

        baseline = list(candidates)

        reranked = (
            retriever.rerank_candidates(
                mention=mention,
                context=context,
                candidates=candidates,
                top_k=TOP_K,
            )
            if candidates
            else []
        )

        rankings = {
            "SBERT": baseline,
            "SBERT + CrossEncoder": reranked,
        }

        baseline_correct = bool(
            baseline and baseline[0]["entity_id"] == gold_id
        )
        reranked_correct = bool(
            reranked and reranked[0]["entity_id"] == gold_id
        )

        fixed += int(not baseline_correct and reranked_correct)
        broken += int(baseline_correct and not reranked_correct)

        for method_name, ranked_candidates in rankings.items():
            ranked_ids = [
                candidate["entity_id"]
                for candidate in ranked_candidates
            ]

            for k in (1, 5, 10):
                if gold_id in ranked_ids[:k]:
                    methods[method_name]["hits"][k] += 1

            methods[method_name]["rr"] += reciprocal_rank(
                ranked_candidates,
                gold_id,
            )

        if index <= 10:
            print(
                f"{index}. {mention!r} | Gold={gold_id} "
                f"| SBERT top-1="
                f"{baseline[0]['entity_id'] if baseline else 'None'} "
                f"| Reranked top-1="
                f"{reranked[0]['entity_id'] if reranked else 'None'}"
            )

        if index % 50 == 0:
            print(f"Processed {index}/{len(sample)}")

    n = len(sample)

    print("\n" + "=" * 60)
    print("PAIRED ENTITY-LINKING COMPARISON")
    print("=" * 60)
    print(f"Examples: {n}")
    print(f"Catalogue coverage: {covered_count / n:.4f}")

    for method_name, values in methods.items():
        print(f"\n{method_name}")
        for k in (1, 5, 10):
            print(
                f"Recall@{k}: "
                f"{values['hits'][k] / n:.4f}"
            )
        print(f"MRR: {values['rr'] / n:.4f}")

    baseline = methods["SBERT"]
    reranked = methods["SBERT + CrossEncoder"]

    print("\nAbsolute changes (reranked minus baseline)")
    for k in (1, 5, 10):
        delta = (
            reranked["hits"][k] - baseline["hits"][k]
        ) / n
        print(f"Recall@{k}: {delta:+.4f}")

    delta_mrr = (
        reranked["rr"] - baseline["rr"]
    ) / n
    print(f"MRR: {delta_mrr:+.4f}")

    print(f"Top-1 errors fixed: {fixed}")
    print(f"Top-1 errors introduced: {broken}")


if __name__ == "__main__":
    main()

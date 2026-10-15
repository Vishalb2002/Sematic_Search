
import random

from datasets import load_dataset
from scipy.stats import binomtest

from src.entity_linking.candidate_retriever import CandidateRetriever


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
TOP_K = 10
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 42


def reciprocal_rank(candidates, gold_id):
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["entity_id"] == gold_id:
            return 1.0 / rank
    return 0.0


def main():
    dataset = load_dataset(DATASET_NAME)
    retriever = CandidateRetriever()

    if retriever.collection.count() == 0:
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

    # Store paired per-example outcomes.
    baseline_rr = []
    reranked_rr = []
    baseline_correct = []
    reranked_correct = []

    total = len(examples)
    print("Test mentions:", total)
    print("Running paired predictions...")

    for index, example in enumerate(examples, start=1):
        candidates = retriever.search(
            mention=example["mention"],
            context=example["context"],
            top_k=TOP_K,
        )

        reranked = (
            retriever.rerank_candidates(
                mention=example["mention"],
                context=example["context"],
                candidates=candidates,
                top_k=TOP_K,
            )
            if candidates else []
        )

        gold_id = example["gold_id"]

        baseline_rr.append(
            reciprocal_rank(candidates, gold_id)
        )
        reranked_rr.append(
            reciprocal_rank(reranked, gold_id)
        )

        baseline_correct.append(
            int(bool(candidates) and
                candidates[0]["entity_id"] == gold_id)
        )
        reranked_correct.append(
            int(bool(reranked) and
                reranked[0]["entity_id"] == gold_id)
        )

        if index % 100 == 0 or index == total:
            print(f"Processed {index}/{total}")

    n = len(baseline_rr)

    observed_mrr_delta = (
        sum(reranked_rr) - sum(baseline_rr)
    ) / n

    # Paired bootstrap: resample test examples with replacement.
    rng = random.Random(RANDOM_SEED)
    bootstrap_deltas = []

    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(n) for _ in range(n)]

        delta = sum(
            reranked_rr[i] - baseline_rr[i]
            for i in indices
        ) / n

        bootstrap_deltas.append(delta)

    bootstrap_deltas.sort()

    lower_index = int(0.025 * BOOTSTRAP_REPLICATES)
    upper_index = int(0.975 * BOOTSTRAP_REPLICATES) - 1

    ci_lower = bootstrap_deltas[lower_index]
    ci_upper = bootstrap_deltas[upper_index]

    # McNemar's exact test for paired top-1 correctness.
    reranker_fixed = sum(
        b == 0 and r == 1
        for b, r in zip(baseline_correct, reranked_correct)
    )
    reranker_broke = sum(
        b == 1 and r == 0
        for b, r in zip(baseline_correct, reranked_correct)
    )

    discordant = reranker_fixed + reranker_broke

    if discordant:
        mcnemar = binomtest(
            reranker_fixed,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )
        p_value = mcnemar.pvalue
    else:
        p_value = 1.0

    baseline_accuracy = sum(baseline_correct) / n
    reranked_accuracy = sum(reranked_correct) / n

    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)
    print("Examples:", n)
    print(f"SBERT top-1 accuracy: {baseline_accuracy:.4f}")
    print(f"Reranked top-1 accuracy: {reranked_accuracy:.4f}")
    print(f"MRR improvement: {observed_mrr_delta:+.4f}")
    print(
        "95% paired-bootstrap CI for MRR change: "
        f"[{ci_lower:+.4f}, {ci_upper:+.4f}]"
    )
    print("Top-1 errors fixed:", reranker_fixed)
    print("Top-1 errors introduced:", reranker_broke)
    print(f"McNemar exact two-sided p-value: {p_value:.6f}")

    if ci_lower > 0 and p_value < 0.05:
        print(
            "\nInterpretation: both tests support an improvement "
            "on this test set under this evaluation protocol."
        )
    else:
        print(
            "\nInterpretation: the evidence does not meet both "
            "predefined statistical criteria. Interpret the "
            "observed improvement cautiously."
        )


if __name__ == "__main__":
    main()


import random
from datasets import load_dataset
from src.entity_linking.candidate_retriever import CandidateRetriever

DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
TOP_K = 10
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 42


def reciprocal_rank(candidates, gold_id):
    for rank, candidate in enumerate(candidates, start=1):
        if str(candidate["entity_id"]) == str(gold_id):
            return 1.0 / rank
    return 0.0


def main():
    dataset = load_dataset(DATASET_NAME)
    retriever = CandidateRetriever()

    if retriever.collection.count() == 0:
        raise RuntimeError("Entity catalogue is empty.")

    # Keep paired results grouped by document.
    document_results = []
    total_mentions = 0

    print("Evaluating test documents...")

    for doc_index, document in enumerate(dataset["test"]):
        text = document["text"]
        mention_results = []

        for entity in document["entities"]:
            gold_id = entity.get("qid")
            title = entity.get("title")

            # Match the filtering used in the earlier evaluation.
            if gold_id is None or title is None:
                continue

            start = entity["start"]
            end = entity["end"]
            mention = text[start:end]
            context = text[max(0, start - 150):min(len(text), end + 150)]

            candidates = retriever.search(
                mention=mention,
                context=context,
                top_k=TOP_K,
            )

            reranked = (
                retriever.rerank_candidates(
                    mention=mention,
                    context=context,
                    candidates=candidates,
                    top_k=TOP_K,
                )
                if candidates else []
            )

            gold_id = str(gold_id)

            baseline_rr = reciprocal_rank(candidates, gold_id)
            reranked_rr = reciprocal_rank(reranked, gold_id)

            baseline_correct = int(
                bool(candidates)
                and str(candidates[0]["entity_id"]) == gold_id
            )
            reranked_correct = int(
                bool(reranked)
                and str(reranked[0]["entity_id"]) == gold_id
            )

            mention_results.append({
                "mrr_delta": reranked_rr - baseline_rr,
                "accuracy_delta": reranked_correct - baseline_correct,
                "baseline_rr": baseline_rr,
                "reranked_rr": reranked_rr,
                "baseline_correct": baseline_correct,
                "reranked_correct": reranked_correct,
            })

        if mention_results:
            document_results.append(mention_results)
            total_mentions += len(mention_results)

        if (doc_index + 1) % 25 == 0 or doc_index + 1 == len(dataset["test"]):
            print(
                f"Processed documents: {doc_index + 1}/"
                f"{len(dataset['test'])}"
            )

    if not document_results:
        raise RuntimeError("No eligible test mentions were evaluated.")

    # Original, non-bootstrap point estimates.
    all_mentions = [
        result
        for document in document_results
        for result in document
    ]

    n = len(all_mentions)
    observed_mrr_delta = sum(
        item["mrr_delta"] for item in all_mentions
    ) / n
    observed_accuracy_delta = sum(
        item["accuracy_delta"] for item in all_mentions
    ) / n

    # Cluster bootstrap: sample documents, not individual mentions.
    rng = random.Random(RANDOM_SEED)
    number_of_documents = len(document_results)
    bootstrap_mrr_deltas = []
    bootstrap_accuracy_deltas = []

    print("\nRunning document-level bootstrap...")

    for iteration in range(BOOTSTRAP_REPLICATES):
        sampled_documents = [
            document_results[rng.randrange(number_of_documents)]
            for _ in range(number_of_documents)
        ]

        # Pool mentions from sampled documents. Documents with more
        # eligible mentions contribute proportionally more, matching
        # the mention-weighted evaluation metric.
        sampled_mentions = [
            result
            for document in sampled_documents
            for result in document
        ]

        denominator = len(sampled_mentions)

        bootstrap_mrr_deltas.append(
            sum(x["mrr_delta"] for x in sampled_mentions) / denominator
        )
        bootstrap_accuracy_deltas.append(
            sum(x["accuracy_delta"] for x in sampled_mentions) / denominator
        )

    def percentile_interval(values):
        values = sorted(values)
        count = len(values)
        lower = values[int(0.025 * count)]
        upper = values[min(int(0.975 * count), count - 1)]
        return lower, upper

    mrr_lower, mrr_upper = percentile_interval(bootstrap_mrr_deltas)
    acc_lower, acc_upper = percentile_interval(bootstrap_accuracy_deltas)

    baseline_mrr = sum(x["baseline_rr"] for x in all_mentions) / n
    reranked_mrr = sum(x["reranked_rr"] for x in all_mentions) / n

    baseline_accuracy = sum(
        x["baseline_correct"] for x in all_mentions
    ) / n
    reranked_accuracy = sum(
        x["reranked_correct"] for x in all_mentions
    ) / n

    print("\n" + "=" * 62)
    print("DOCUMENT-LEVEL BOOTSTRAP RESULTS")
    print("=" * 62)
    print("Eligible documents:", number_of_documents)
    print("Eligible linked mentions:", total_mentions)
    print("Bootstrap replicates:", BOOTSTRAP_REPLICATES)
    print()
    print(f"SBERT MRR:                 {baseline_mrr:.4f}")
    print(f"SBERT + CrossEncoder MRR:  {reranked_mrr:.4f}")
    print(f"Observed MRR change:       {observed_mrr_delta:+.4f}")
    print(
        "95% document-bootstrap CI for MRR change: "
        f"[{mrr_lower:+.4f}, {mrr_upper:+.4f}]"
    )
    print()
    print(f"SBERT top-1 accuracy:      {baseline_accuracy:.4f}")
    print(f"Reranked top-1 accuracy:   {reranked_accuracy:.4f}")
    print(f"Observed accuracy change:  {observed_accuracy_delta:+.4f}")
    print(
        "95% document-bootstrap CI for accuracy change: "
        f"[{acc_lower:+.4f}, {acc_upper:+.4f}]"
    )
    print("=" * 62)

    if mrr_lower > 0:
        print("MRR conclusion: CI supports a positive improvement.")
    elif mrr_upper < 0:
        print("MRR conclusion: CI supports a negative change.")
    else:
        print("MRR conclusion: CI includes zero; improvement is uncertain.")


if __name__ == "__main__":
    main()

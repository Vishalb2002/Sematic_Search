import csv
import json
import time
from pathlib import Path

from datasets import load_dataset

from src.entity_linking.candidate_retriever import CandidateRetriever
from src.retrieval.cross_encoder import CrossEncoderModel


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
SPLIT = "validation"

TOP_K = 10
RERANK_BATCH_SIZE = 256

OUTPUT_DIR = Path("outputs/entity_linking_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_context(text, start, end, window=150):
    """Return the local text surrounding an entity mention."""
    return (
        text[max(0, start - window):min(len(text), end + window)]
        .replace("\n", " ")
        .strip()
    )


def ranking_metrics(examples, ranking_key, denominator=None):
    """Compute recall@k, MRR, and top-1 accuracy."""
    if denominator is None:
        denominator = len(examples)

    metrics = {}

    for k in (1, 5, 10):
        hits = 0

        for example in examples:
            ranked_ids = [
                str(candidate["entity_id"])
                for candidate in example[ranking_key][:k]
            ]

            if example["gold_id"] in ranked_ids:
                hits += 1

        metrics[f"recall@{k}"] = (
            hits / denominator if denominator else 0.0
        )

    reciprocal_rank_sum = 0.0
    top1_correct = 0

    for example in examples:
        ranked_ids = [
            str(candidate["entity_id"])
            for candidate in example[ranking_key]
        ]

        if ranked_ids and ranked_ids[0] == example["gold_id"]:
            top1_correct += 1

        try:
            rank = ranked_ids.index(example["gold_id"]) + 1
            reciprocal_rank_sum += 1.0 / rank
        except ValueError:
            pass

    metrics["mrr"] = (
        reciprocal_rank_sum / denominator if denominator else 0.0
    )
    metrics["top1_accuracy"] = (
        top1_correct / denominator if denominator else 0.0
    )

    return metrics


def main():
    start_time = time.time()

    print(f"Loading AIDA dataset: {SPLIT}")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    retriever = CandidateRetriever()
    collection = retriever.collection

    catalogue_size = collection.count()

    if catalogue_size == 0:
        raise RuntimeError(
            "The entity catalogue is empty. Build it first with:\n"
            "python -m src.entity_linking.build_dataset_catalogue"
        )

    print(f"Catalogue size: {catalogue_size}")
    print(f"Validation documents: {len(dataset)}")

    examples = []
    total_annotations = 0
    skipped_unlinked = 0
    example_number = 0

    # Retrieve candidates for every linked validation mention.
    for doc_index, document in enumerate(dataset):
        text = document["text"]

        for annotation in document["entities"]:
            total_annotations += 1

            gold_id = annotation.get("qid")
            gold_title = annotation.get("title")

            # This experiment evaluates linking for known gold entities.
            # NIL mentions are excluded and must be evaluated separately.
            if gold_id is None or gold_title is None:
                skipped_unlinked += 1
                continue

            start = annotation["start"]
            end = annotation["end"]
            mention = text[start:end].strip()

            if not mention:
                continue

            context = get_context(text, start, end)

            candidates = retriever.search(
                mention=mention,
                context=context,
                top_k=TOP_K,
                # Do not filter by type: dataset type labels and catalogue
                # types may be noisy or inconsistently mapped.
            )

            example_number += 1

            examples.append({
                "example_id": example_number,
                "document_index": doc_index,
                "mention": mention,
                "context": context,
                "gold_id": str(gold_id),
                "gold_title": str(gold_title).replace("_", " "),
                "entity_type": annotation.get("tag"),
                "retrieved": candidates,
            })

        if (doc_index + 1) % 25 == 0:
            print(
                f"Processed {doc_index + 1}/{len(dataset)} documents; "
                f"linked mentions collected: {len(examples)}"
            )

    if not examples:
        raise RuntimeError("No linked validation mentions were found.")

    # Candidate coverage: can the catalogue retrieve the gold entity at all?
    covered_examples = [
        example for example in examples
        if any(
            str(candidate["entity_id"]) == example["gold_id"]
            for candidate in example["retrieved"]
        )
    ]

    coverage = len(covered_examples) / len(examples)

    retrieval_metrics = ranking_metrics(
        examples, "retrieved"
    )

    # Rerank the SAME candidate pool using the existing CrossEncoder.
    # Batch scoring avoids loading/calling the model separately per mention.
    print("\nLoading CrossEncoder...")
    cross_encoder = CrossEncoderModel.get_model()

    pair_records = []

    for example_index, example in enumerate(examples):
        query = (
            f"Entity mention: {example['mention']}\n"
            f"Context: {example['context']}"
        )

        for candidate_index, candidate in enumerate(example["retrieved"]):
            candidate_text = (
                f"Entity: {candidate['name']}. "
                f"Type: {candidate['entity_type']}. "
                f"Description: {candidate['description']}. "
                f"Aliases: {candidate.get('aliases', '')}"
            )

            pair_records.append({
                "example_index": example_index,
                "candidate_index": candidate_index,
                "pair": (query, candidate_text),
            })

    print(f"Scoring {len(pair_records)} mention-candidate pairs...")

    scores = []

    for start in range(0, len(pair_records), RERANK_BATCH_SIZE):
        batch = pair_records[start:start + RERANK_BATCH_SIZE]
        batch_pairs = [record["pair"] for record in batch]

        batch_scores = cross_encoder.predict(batch_pairs)
        scores.extend(float(score) for score in batch_scores)

        completed = min(start + len(batch), len(pair_records))
        if completed % 2000 < RERANK_BATCH_SIZE:
            print(f"Scored {completed}/{len(pair_records)} pairs")

    for record, score in zip(pair_records, scores):
        candidate = examples[record["example_index"]]["retrieved"][
            record["candidate_index"]
        ]
        candidate["rerank_score"] = score

    for example in examples:
        example["reranked"] = sorted(
            example["retrieved"],
            key=lambda candidate: candidate["rerank_score"],
            reverse=True,
        )

    reranked_metrics = ranking_metrics(examples, "reranked")

    # Conditional ranking metrics measure performance only where the
    # gold entity appears in the retrieved top-10 candidates.
    retrieval_covered_metrics = ranking_metrics(
        covered_examples, "retrieved"
    )
    reranked_covered_metrics = ranking_metrics(
        covered_examples, "reranked"
    )

    # Count cases where reranking fixed or introduced a top-1 error.
    retrieval_correct = 0
    reranked_correct = 0
    fixed = 0
    introduced = 0

    for example in examples:
        gold_id = example["gold_id"]

        before_correct = (
            bool(example["retrieved"])
            and str(example["retrieved"][0]["entity_id"]) == gold_id
        )
        after_correct = (
            bool(example["reranked"])
            and str(example["reranked"][0]["entity_id"]) == gold_id
        )

        retrieval_correct += int(before_correct)
        reranked_correct += int(after_correct)

        if not before_correct and after_correct:
            fixed += 1
        elif before_correct and not after_correct:
            introduced += 1

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "catalogue_size": catalogue_size,
        "validation_documents": len(dataset),
        "total_annotations": total_annotations,
        "linked_mentions_evaluated": len(examples),
        "unlinked_annotations_excluded": skipped_unlinked,
        "catalogue_coverage_at_10": coverage,
        "retrieval_metrics_all_linked_mentions": retrieval_metrics,
        "reranked_metrics_all_linked_mentions": reranked_metrics,
        "retrieval_metrics_covered_mentions": retrieval_covered_metrics,
        "reranked_metrics_covered_mentions": reranked_covered_metrics,
        "top1_correct_before_reranking": retrieval_correct,
        "top1_correct_after_reranking": reranked_correct,
        "top1_errors_fixed": fixed,
        "top1_errors_introduced": introduced,
        "net_top1_correct_change": reranked_correct - retrieval_correct,
        "runtime_seconds": round(time.time() - start_time, 2),
        "note": (
            "Gold-mention entity-linking evaluation only. "
            "Unlinked/NIL annotations are excluded. "
            "CrossEncoder scores are not probabilities. "
            "Catalogue coverage is limited to the retrieved top-10."
        ),
    }

    summary_path = OUTPUT_DIR / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # Save per-mention rankings for error analysis.
    details_path = OUTPUT_DIR / "validation_predictions.json"
    with details_path.open("w", encoding="utf-8") as file:
        json.dump(
            [
                {
                    "example_id": example["example_id"],
                    "document_index": example["document_index"],
                    "mention": example["mention"],
                    "context": example["context"],
                    "gold_id": example["gold_id"],
                    "gold_title": example["gold_title"],
                    "entity_type": example["entity_type"],
                    "retrieved": example["retrieved"],
                    "reranked": example["reranked"],
                }
                for example in examples
            ],
            file,
            indent=2,
        )

    # Also write a compact CSV for spreadsheet analysis.
    csv_path = OUTPUT_DIR / "validation_predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "example_id",
                "document_index",
                "mention",
                "context",
                "gold_id",
                "gold_title",
                "entity_type",
                "retrieval_top1_id",
                "retrieval_top1_name",
                "rerank_top1_id",
                "rerank_top1_name",
                "gold_in_top10",
                "retrieval_top1_correct",
                "rerank_top1_correct",
            ],
        )
        writer.writeheader()

        for example in examples:
            retrieved = example["retrieved"]
            reranked = example["reranked"]
            gold_id = example["gold_id"]

            writer.writerow({
                "example_id": example["example_id"],
                "document_index": example["document_index"],
                "mention": example["mention"],
                "context": example["context"],
                "gold_id": gold_id,
                "gold_title": example["gold_title"],
                "entity_type": example["entity_type"],
                "retrieval_top1_id": (
                    retrieved[0]["entity_id"] if retrieved else ""
                ),
                "retrieval_top1_name": (
                    retrieved[0]["name"] if retrieved else ""
                ),
                "rerank_top1_id": (
                    reranked[0]["entity_id"] if reranked else ""
                ),
                "rerank_top1_name": (
                    reranked[0]["name"] if reranked else ""
                ),
                "gold_in_top10": int(any(
                    str(candidate["entity_id"]) == gold_id
                    for candidate in retrieved
                )),
                "retrieval_top1_correct": int(
                    bool(retrieved)
                    and str(retrieved[0]["entity_id"]) == gold_id
                ),
                "rerank_top1_correct": int(
                    bool(reranked)
                    and str(reranked[0]["entity_id"]) == gold_id
                ),
            })

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Linked mentions evaluated: {len(examples)}")
    print(f"Catalogue coverage @10: {coverage:.4f}")

    print("\nSBERT retrieval — all linked mentions")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nSBERT + CrossEncoder — all linked mentions")
    for metric, value in reranked_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nSBERT retrieval — catalogue-covered mentions")
    for metric, value in retrieval_covered_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nSBERT + CrossEncoder — catalogue-covered mentions")
    for metric, value in reranked_covered_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTop-1 comparison")
    print(f"  Correct before reranking: {retrieval_correct}")
    print(f"  Correct after reranking:  {reranked_correct}")
    print(f"  Errors fixed:             {fixed}")
    print(f"  Errors introduced:        {introduced}")
    print(f"  Net change:               {reranked_correct - retrieval_correct:+d}")

    print("\nSaved results:")
    print(f"  {summary_path}")
    print(f"  {details_path}")
    print(f"  {csv_path}")
    print(f"Runtime: {summary['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()

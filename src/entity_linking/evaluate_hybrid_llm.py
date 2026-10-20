import json
import time
from pathlib import Path

from datasets import load_dataset

from src.entity_linking.hybrid_candidate_retriever import (
    HybridCandidateRetriever,
)
from src.entity_linking.llm_entity_linker import LLMEntityLinker


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
SPLIT = "validation"

# Start small; increase only after confirming the results.
MAX_MENTIONS = 20
TOP_K = 10

OUTPUT_DIR = Path("outputs/entity_linking_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_context(text, start, end, window=150):
    """Return the mention with its surrounding context."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


def calculate_metrics(records, prediction_key):
    """Calculate end-to-end top-1 accuracy and MRR."""
    if not records:
        return {
            "examples": 0,
            "accuracy": 0.0,
            "mrr": 0.0,
        }

    correct = 0
    reciprocal_rank_sum = 0

    for record in records:
        gold_id = record["gold_id"]
        prediction = record[prediction_key]

        if prediction == gold_id:
            correct += 1

        # This evaluation uses a single selected entity.
        if prediction == gold_id:
            reciprocal_rank_sum += 1

    total = len(records)

    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total,
        "mrr": reciprocal_rank_sum / total,
    }


def main():
    print("Loading validation dataset...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    print("Loading hybrid candidate retriever...")
    retriever = HybridCandidateRetriever()

    print("Loading LLM entity linker...")
    linker = LLMEntityLinker(model_name="llama3.2:3b")

    records = []
    linked_mentions_seen = 0

    print(f"\nEvaluating up to {MAX_MENTIONS} linked mentions...")

    start_time = time.time()

    for document_index, document in enumerate(dataset):
        text = document["text"]

        for annotation in document["entities"]:
            gold_id = annotation.get("qid")
            gold_title = annotation.get("title")

            # Match the linked-mention protocol of the existing evaluator.
            if not gold_id or not gold_title:
                continue

            start = annotation.get("start")
            end = annotation.get("end")

            if start is None or end is None:
                raise ValueError(
                    "Entity annotation has no start/end offsets. "
                    f"Available keys: {list(annotation.keys())}"
                )

            mention = text[start:end]

            if not mention.strip():
                continue

            context = get_context(text, start, end)

            candidates = retriever.search(
                mention=mention,
                context=context,
                top_k=TOP_K,
            )

          
            candidate_ids = [
                str(candidate["entity_id"])
                for candidate in candidates
            ]

            candidate_details = [
                {
                    "entity_id": str(candidate.get("entity_id", "")),
                    "name": candidate.get("name", ""),
                    "entity_type": candidate.get("entity_type", ""),
                    "description": candidate.get("description", ""),
                    "aliases": candidate.get("aliases", []),
                    "distance": candidate.get("distance"),
                }
                for candidate in candidates
            ]


            # Retrieval-only baseline: first hybrid candidate.
            retrieval_prediction = (
                candidate_ids[0] if candidate_ids else "NIL"
            )

            # LLM selection uses exactly the same candidate list.
            llm_result = linker.link(
                mention=mention,
                context=context,
                candidates=candidates,
            )

            llm_prediction = llm_result["entity_id"]
            gold_id = str(gold_id)

            records.append({
                "document_index": document_index,
                "mention": mention,
                "context": context,
                "gold_id": gold_id,
                "gold_title": gold_title,
                "candidate_ids": candidate_ids,
                "candidate_details": candidate_details,
                "gold_in_candidates": gold_id in candidate_ids,
                "retrieval_prediction": retrieval_prediction,
                "llm_prediction": llm_prediction,
                "llm_reason": llm_result["reason"],
                "llm_valid_selection": llm_result["valid_selection"],
                "llm_raw_response": llm_result["raw_response"],
            })

            linked_mentions_seen += 1

            print(
                f"[{linked_mentions_seen}/{MAX_MENTIONS}] "
                f"Mention={mention!r} | "
                f"Gold={gold_id} | "
                f"Retrieval={retrieval_prediction} | "
                f"LLM={llm_prediction}"
            )

            if linked_mentions_seen >= MAX_MENTIONS:
                break

        if linked_mentions_seen >= MAX_MENTIONS:
            break

    if not records:
        raise RuntimeError(
            "No linked mentions were evaluated. "
            "Check the dataset annotation format."
        )

    # Overall end-to-end metrics: missing candidates count as errors.
    retrieval_metrics = calculate_metrics(
        records, "retrieval_prediction"
    )
    llm_metrics = calculate_metrics(records, "llm_prediction")

    # Diagnostic metric: only mentions whose gold entity was retrieved.
    covered_records = [
        record for record in records
        if record["gold_in_candidates"]
    ]

    retrieval_covered_metrics = calculate_metrics(
        covered_records, "retrieval_prediction"
    )
    llm_covered_metrics = calculate_metrics(
        covered_records, "llm_prediction"
    )

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "model": "llama3.2:3b",
        "sample_size": len(records),
        "candidate_limit": TOP_K,
        "candidate_coverage": (
            sum(r["gold_in_candidates"] for r in records)
            / len(records)
        ),
        "retrieval_only": retrieval_metrics,
        "hybrid_retrieval_plus_llm": llm_metrics,
        "retrieval_only_on_covered_mentions": retrieval_covered_metrics,
        "llm_on_covered_mentions": llm_covered_metrics,
        "valid_llm_selection_rate": (
            sum(r["llm_valid_selection"] for r in records)
            / len(records)
        ),
        "runtime_seconds": round(time.time() - start_time, 2),
        "note": (
            "Small exploratory validation sample; not a final "
            "research result. Only linked gold mentions are evaluated. "
            "Covered-mention accuracy is conditional on the gold entity "
            "appearing in the retrieved candidates."
        ),
    }

    results_path = OUTPUT_DIR / "hybrid_llm_sample_predictions.json"
    summary_path = OUTPUT_DIR / "hybrid_llm_sample_summary.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n========== SAMPLE EVALUATION ==========")
    print(json.dumps(summary, indent=2))
    print("\nSaved predictions:", results_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
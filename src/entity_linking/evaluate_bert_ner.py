
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import pipeline


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
MODEL_NAME = "dslim/bert-base-NER"

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

OUTPUT_DIR = Path("outputs/bert_ner")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(gold_entities, predicted_entities):
    """
    Entity format: (start, end, entity_type).
    Strict matching requires exact span AND exact entity type.
    """
    gold_set = set(gold_entities)
    pred_set = set(predicted_entities)

    true_positive = len(gold_set & pred_set)
    false_positive = len(pred_set - gold_set)
    false_negative = len(gold_set - pred_set)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "gold": len(gold_set),
        "predicted": len(pred_set),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def span_only_metrics(gold_entities, predicted_entities):
    """Evaluate exact boundaries, ignoring entity types."""
    gold_spans = {(start, end) for start, end, _ in gold_entities}
    pred_spans = {(start, end) for start, end, _ in predicted_entities}
    return calculate_metrics(
        [(s, e, "ENTITY") for s, e in gold_spans],
        [(s, e, "ENTITY") for s, e in pred_spans],
    )


def main():
    print("=" * 65)
    print("BERT NAMED ENTITY RECOGNITION EVALUATION")
    print("=" * 65)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'CUDA GPU' if device == 0 else 'CPU'}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")

    print("\nLoading validation split...")
    dataset = load_dataset(DATASET_NAME, split="validation")
    print(f"Validation documents: {len(dataset)}")

    print("\nLoading pretrained NER model...")
    ner_pipeline = pipeline(
        "token-classification",
        model=MODEL_NAME,
        aggregation_strategy="simple",
        device=device,
        stride=64,
    )

    all_gold = []
    all_predicted = []
    per_type_gold = defaultdict(list)
    per_type_predicted = defaultdict(list)
    examples = []
    total_inference_seconds = 0.0
    total_characters = 0

    print("\nEvaluating documents...")

    for index, document in enumerate(dataset):
        text = document["text"]
        raw_entities = document["entities"]

        # AIDA stores entity offsets as start-inclusive/end-exclusive.
        # Keep only valid annotations with recognized entity types.
        gold = []
        for entity in raw_entities:
            start = int(entity["start"])
            end = int(entity["end"])
            label = entity["tag"]

            if (
                label in ENTITY_TYPES
                and 0 <= start < end <= len(text)
            ):
                gold.append((start, end, label))

        start_time = time.perf_counter()

        try:
            outputs = ner_pipeline(text)
        except Exception as exc:
            print(
                f"\nERROR on document "
                f"{document.get('document_id', index)}: {exc}"
            )
            raise

        total_inference_seconds += time.perf_counter() - start_time
        total_characters += len(text)

        predicted = []
        for item in outputs:
            label = item.get("entity_group", item.get("entity", ""))
            label = label.replace("B-", "").replace("I-", "")

            start = item.get("start")
            end = item.get("end")

            if start is None or end is None:
                continue

            start, end = int(start), int(end)

            if (
                label in ENTITY_TYPES
                and 0 <= start < end <= len(text)
            ):
                predicted.append((start, end, label))

        # Remove duplicate predictions, if any.
        gold = sorted(set(gold))
        predicted = sorted(set(predicted))

        all_gold.extend(gold)
        all_predicted.extend(predicted)

        for label in ENTITY_TYPES:
            per_type_gold[label].extend(
                entity for entity in gold if entity[2] == label
            )
            per_type_predicted[label].extend(
                entity for entity in predicted if entity[2] == label
            )

        if index < 5:
            examples.append({
                "document_id": document.get("document_id", index),
                "text_preview": text[:500],
                "gold_entities": [
                    {
                        "text": text[s:e],
                        "type": label,
                        "start": s,
                        "end": e,
                    }
                    for s, e, label in gold
                ],
                "predicted_entities": [
                    {
                        "text": text[s:e],
                        "type": label,
                        "start": s,
                        "end": e,
                    }
                    for s, e, label in predicted
                ],
            })

        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            print(f"Processed {index + 1}/{len(dataset)} documents")

    # Span-only evaluation: boundaries must match; labels may differ.
    span_gold = [(s, e, "ENTITY") for s, e, _ in all_gold]
    span_pred = [(s, e, "ENTITY") for s, e, _ in all_predicted]

    results = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "split": "validation",
        "documents": len(dataset),
        "device": "CUDA" if device == 0 else "CPU",
        "inference_seconds": round(total_inference_seconds, 3),
        "characters_processed": total_characters,
        "documents_per_second": round(
            len(dataset) / total_inference_seconds, 3
        ) if total_inference_seconds else None,
        "span_only": calculate_metrics(span_gold, span_pred),
        "strict_span_and_type": calculate_metrics(
            all_gold, all_predicted
        ),
        "per_type": {},
        "notes": [
            "Pretrained model; not fine-tuned on AIDA.",
            "Validation split only; test split was not evaluated.",
            "Strict matching requires exact start, end, and entity type.",
            "Inference time excludes initial model download and loading.",
        ],
    }

    for label in ENTITY_TYPES:
        results["per_type"][label] = calculate_metrics(
            per_type_gold[label],
            per_type_predicted[label],
        )

    report_path = OUTPUT_DIR / "bert_ner_validation_results.json"
    examples_path = OUTPUT_DIR / "bert_ner_validation_examples.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with examples_path.open("w", encoding="utf-8") as file:
        json.dump(examples, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)

    for metric_name, metrics in [
        ("SPAN ONLY", results["span_only"]),
        ("STRICT SPAN + TYPE", results["strict_span_and_type"]),
    ]:
        print(f"\n{metric_name}")
        print(f"  Gold entities: {metrics['gold']}")
        print(f"  Predicted entities: {metrics['predicted']}")
        print(f"  True positives: {metrics['true_positive']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

    print("\nPER-ENTITY RESULTS")
    print("-" * 65)
    print(f"{'Type':<10}{'Precision':>12}{'Recall':>12}{'F1':>12}")

    for label in ENTITY_TYPES:
        metrics = results["per_type"][label]
        print(
            f"{label:<10}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
        )

    print("\nRUNTIME")
    print(f"  Total inference time: {total_inference_seconds:.2f} seconds")
    print(f"  Documents/second: {results['documents_per_second']}")
    print(f"\nResults saved to: {report_path}")
    print(f"Examples saved to: {examples_path}")


if __name__ == "__main__":
    main()

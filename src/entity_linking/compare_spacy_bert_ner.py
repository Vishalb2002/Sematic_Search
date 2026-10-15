
import json
import time
from pathlib import Path
from collections import defaultdict

import spacy
import torch
from datasets import load_dataset
from transformers import pipeline


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
BERT_MODEL = "dslim/bert-base-NER"
SPACY_MODEL = "en_core_web_sm"

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

# Explicit mapping from spaCy labels to the AIDA annotation scheme.
SPACY_LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "PRODUCT": "MISC",
    "NORP": "MISC",
}

OUTPUT_DIR = Path("outputs/ner_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_metrics(gold, predicted):
    gold_set = set(gold)
    pred_set = set(predicted)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "gold": len(gold_set),
        "predicted": len(pred_set),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(gold, predicted):
    # Span-only matching ignores entity types.
    gold_spans = {(s, e) for s, e, _ in gold}
    pred_spans = {(s, e) for s, e, _ in predicted}

    span_metrics = get_metrics(
        [(s, e, "ENTITY") for s, e in gold_spans],
        [(s, e, "ENTITY") for s, e in pred_spans],
    )

    # Strict matching requires exact start, end, and entity type.
    strict_metrics = get_metrics(gold, predicted)

    per_type = {}
    for label in ENTITY_TYPES:
        type_gold = [x for x in gold if x[2] == label]
        type_pred = [x for x in predicted if x[2] == label]
        per_type[label] = get_metrics(type_gold, type_pred)

    return {
        "span_only": span_metrics,
        "strict_span_and_type": strict_metrics,
        "per_type": per_type,
    }


def main():
    print("Loading AIDA validation split...")
    dataset = load_dataset(DATASET_NAME, split="validation")
    print(f"Validation documents: {len(dataset)}")

    print("\nLoading spaCy model...")
    nlp = spacy.load(SPACY_MODEL)

    device = 0 if torch.cuda.is_available() else -1
    print("\nLoading BERT NER model...")
    bert_ner = pipeline(
        "token-classification",
        model=BERT_MODEL,
        aggregation_strategy="simple",
        device=device,
        stride=64,
    )

    gold_all = []
    spacy_all = []
    bert_all = []

    spacy_seconds = 0.0
    bert_seconds = 0.0
    total_characters = 0
    duplicate_gold_spans = 0
    invalid_gold_annotations = 0
    spacy_unmapped_count = 0

    print("\nEvaluating both models on identical documents...")

    for i, document in enumerate(dataset):
        text = document["text"]
        total_characters += len(text)

        gold = []
        seen_gold_spans = set()

        for entity in document["entities"]:
            start = int(entity["start"])
            end = int(entity["end"])
            label = entity["tag"]

            if (
                label not in ENTITY_TYPES
                or not (0 <= start < end <= len(text))
            ):
                invalid_gold_annotations += 1
                continue

            if (start, end) in seen_gold_spans:
                duplicate_gold_spans += 1
            seen_gold_spans.add((start, end))
            gold.append((start, end, label))

        # spaCy prediction
        start_time = time.perf_counter()
        spacy_doc = nlp(text)
        spacy_seconds += time.perf_counter() - start_time

        spacy_predictions = []

        for ent in spacy_doc.ents:
            mapped_label = SPACY_LABEL_MAP.get(ent.label_)

            if mapped_label is None:
                # Keep unmapped predictions visible as strict false positives.
                mapped_label = f"UNMAPPED:{ent.label_}"
                spacy_unmapped_count += 1

            spacy_predictions.append(
                (ent.start_char, ent.end_char, mapped_label)
            )

        # BERT prediction
        start_time = time.perf_counter()
        bert_outputs = bert_ner(text)
        bert_seconds += time.perf_counter() - start_time

        bert_predictions = []

        for ent in bert_outputs:
            label = ent.get("entity_group", ent.get("entity", ""))
            label = label.replace("B-", "").replace("I-", "")
            start = ent.get("start")
            end = ent.get("end")

            if start is None or end is None:
                continue

            start, end = int(start), int(end)

            if (
                label in ENTITY_TYPES
                and 0 <= start < end <= len(text)
            ):
                bert_predictions.append((start, end, label))

        gold_all.extend(gold)
        spacy_all.extend(spacy_predictions)
        bert_all.extend(bert_predictions)

        if (i + 1) % 20 == 0 or i + 1 == len(dataset):
            print(f"Processed {i + 1}/{len(dataset)}")

    results = {
        "dataset": DATASET_NAME,
        "split": "validation",
        "documents": len(dataset),
        "characters": total_characters,
        "annotation_diagnostics": {
            "duplicate_gold_spans": duplicate_gold_spans,
            "invalid_or_unrecognized_gold_annotations":
                invalid_gold_annotations,
            "spacy_unmapped_predictions": spacy_unmapped_count,
        },
        "spacy": evaluate(gold_all, spacy_all),
        "bert": evaluate(gold_all, bert_all),
        "runtime_seconds": {
            "spacy": round(spacy_seconds, 3),
            "bert": round(bert_seconds, 3),
        },
        "methodology": {
            "span_only": "Exact start/end offsets, ignoring type.",
            "strict": "Exact start/end offsets and exact type.",
            "unmapped_spacy_labels":
                "Counted as strict false positives, not silently discarded.",
            "gold_duplicates":
                "Strict matching uses unique (start, end, type) triples; "
                "span-only matching uses unique spans.",
        },
    }

    output_file = OUTPUT_DIR / "spacy_vs_bert_validation.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for model_name in ("spacy", "bert"):
        model_results = results[model_name]

        print("\n" + "=" * 60)
        print(model_name.upper())

        for metric_name in ("span_only", "strict_span_and_type"):
            m = model_results[metric_name]
            print(f"\n{metric_name}")
            print(f"  Gold:      {m['gold']}")
            print(f"  Predicted: {m['predicted']}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1:        {m['f1']:.4f}")

        print("\nPer-type strict metrics:")
        for label, m in model_results["per_type"].items():
            print(
                f"  {label:<5} "
                f"P={m['precision']:.4f} "
                f"R={m['recall']:.4f} "
                f"F1={m['f1']:.4f}"
            )

    print("\n" + "=" * 60)
    print("RUNTIME")
    print(f"spaCy: {spacy_seconds:.2f} seconds")
    print(f"BERT:  {bert_seconds:.2f} seconds")

    print("\nANNOTATION DIAGNOSTICS")
    print(json.dumps(results["annotation_diagnostics"], indent=2))
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()

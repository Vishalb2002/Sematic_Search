
import json
import time
from pathlib import Path
from collections import Counter

import spacy
import torch
from datasets import load_dataset
from transformers import pipeline


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
BERT_MODEL = "dslim/bert-base-NER"
SPACY_MODEL = "en_core_web_sm"

LABELS = ["PER", "ORG", "LOC", "MISC"]

# Conservative, explicit mapping into the AIDA label scheme.
SPACY_LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "NORP": "MISC",
}

OUTPUT_DIR = Path("outputs/ner_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def metrics(gold, predicted):
    """Calculate micro-averaged metrics from document-aware tuples."""
    gold_set = set(gold)
    predicted_set = set(predicted)

    tp = len(gold_set & predicted_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "gold": len(gold_set),
        "predicted": len(predicted_set),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(gold, predicted):
    # Tuples contain (document_id, start, end, label).
    gold_spans = {(d, s, e, "ENTITY") for d, s, e, _ in gold}
    pred_spans = {(d, s, e, "ENTITY") for d, s, e, _ in predicted}

    result = {
        "span_only": metrics(gold_spans, pred_spans),
        "strict_span_and_type": metrics(gold, predicted),
        "per_type": {},
    }

    for label in LABELS:
        label_gold = [x for x in gold if x[3] == label]
        label_pred = [x for x in predicted if x[3] == label]
        result["per_type"][label] = metrics(label_gold, label_pred)

    return result


def entity_text(text, start, end):
    return text[start:end]


def main():
    print("Loading AIDA validation split...")
    dataset = load_dataset(DATASET_NAME, split="validation")
    print(f"Documents: {len(dataset)}")

    print("\nLoading spaCy...")
    nlp = spacy.load(SPACY_MODEL)

    print("\nLoading BERT...")
    device = 0 if torch.cuda.is_available() else -1
    bert = pipeline(
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
    unmapped_labels = Counter()
    invalid_gold = 0
    duplicate_gold = 0
    disagreements = []

    for doc_id, document in enumerate(dataset):
        text = document["text"]

        # Gold annotations retain document identity.
        gold = []
        seen = set()

        for entity in document["entities"]:
            start = int(entity["start"])
            end = int(entity["end"])
            label = entity["tag"]

            if (
                label not in LABELS
                or not (0 <= start < end <= len(text))
            ):
                invalid_gold += 1
                continue

            item = (doc_id, start, end, label)

            if item in seen:
                duplicate_gold += 1
            seen.add(item)
            gold.append(item)

        # spaCy
        start_time = time.perf_counter()
        spacy_doc = nlp(text)
        spacy_seconds += time.perf_counter() - start_time

        spacy_predictions = []

        for ent in spacy_doc.ents:
            label = SPACY_LABEL_MAP.get(ent.label_)

            if label is None:
                unmapped_labels[ent.label_] += 1
                continue

            spacy_predictions.append(
                (doc_id, ent.start_char, ent.end_char, label)
            )

        # BERT
        start_time = time.perf_counter()
        bert_outputs = bert(text)
        bert_seconds += time.perf_counter() - start_time

        bert_predictions = []

        for ent in bert_outputs:
            label = ent.get("entity_group", ent.get("entity", ""))
            label = label.replace("B-", "").replace("I-", "")

            start = ent.get("start")
            end = ent.get("end")

            if (
                label in LABELS
                and start is not None
                and end is not None
                and 0 <= int(start) < int(end) <= len(text)
            ):
                bert_predictions.append(
                    (doc_id, int(start), int(end), label)
                )

        gold = sorted(set(gold))
        spacy_predictions = sorted(set(spacy_predictions))
        bert_predictions = sorted(set(bert_predictions))

        gold_all.extend(gold)
        spacy_all.extend(spacy_predictions)
        bert_all.extend(bert_predictions)

        # Save a few examples where model predictions differ.
        spacy_set = set(spacy_predictions)
        bert_set = set(bert_predictions)

        if spacy_set != bert_set and len(disagreements) < 25:
            disagreements.append({
                "document_index": doc_id,
                "text_preview": text[:600],
                "gold": [
                    {
                        "text": entity_text(text, s, e),
                        "type": label,
                        "start": s,
                        "end": e,
                    }
                    for _, s, e, label in gold
                ],
                "spacy_only": [
                    {
                        "text": entity_text(text, s, e),
                        "type": label,
                        "start": s,
                        "end": e,
                    }
                    for _, s, e, label in sorted(spacy_set - bert_set)
                ],
                "bert_only": [
                    {
                        "text": entity_text(text, s, e),
                        "type": label,
                        "start": s,
                        "end": e,
                    }
                    for _, s, e, label in sorted(bert_set - spacy_set)
                ],
            })

        if (doc_id + 1) % 20 == 0 or doc_id + 1 == len(dataset):
            print(f"Processed {doc_id + 1}/{len(dataset)}")

    results = {
        "dataset": DATASET_NAME,
        "split": "validation",
        "documents": len(dataset),
        "models": {
            "spacy": SPACY_MODEL,
            "bert": BERT_MODEL,
        },
        "protocol": {
            "span_only": "Exact document ID and start/end offsets; type ignored.",
            "strict": "Exact document ID, offsets, and entity type.",
            "spacy_mapping": SPACY_LABEL_MAP,
            "unmapped_spacy_policy":
                "Excluded from target-class scoring and counted separately.",
            "aggregation":
                "Micro-averaged over document-aware entity tuples.",
        },
        "diagnostics": {
            "invalid_or_unrecognized_gold_annotations": invalid_gold,
            "duplicate_gold_annotations": duplicate_gold,
            "unmapped_spacy_label_counts": dict(unmapped_labels),
            "total_unmapped_spacy_predictions": sum(unmapped_labels.values()),
        },
        "spacy": evaluate(gold_all, spacy_all),
        "bert": evaluate(gold_all, bert_all),
        "runtime_seconds": {
            "spacy": round(spacy_seconds, 3),
            "bert": round(bert_seconds, 3),
        },
    }

    results_path = OUTPUT_DIR / "spacy_vs_bert_validation_v2.json"
    examples_path = OUTPUT_DIR / "ner_model_disagreements.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(disagreements, f, indent=2, ensure_ascii=False)

    for model_name in ("spacy", "bert"):
        print("\n" + "=" * 60)
        print(model_name.upper())
        model_result = results[model_name]

        for metric_name in ("span_only", "strict_span_and_type"):
            m = model_result[metric_name]
            print(f"\n{metric_name}")
            print(f"  Gold:      {m['gold']}")
            print(f"  Predicted: {m['predicted']}")
            print(f"  TP:        {m['true_positive']}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1:        {m['f1']:.4f}")

        print("\nPer-type strict F1:")
        for label, m in model_result["per_type"].items():
            print(
                f"  {label}: F1={m['f1']:.4f}, "
                f"P={m['precision']:.4f}, R={m['recall']:.4f}"
            )

    print("\nRUNTIME")
    print(f"spaCy: {spacy_seconds:.2f} seconds")
    print(f"BERT:  {bert_seconds:.2f} seconds")

    print("\nDIAGNOSTICS")
    print(json.dumps(results["diagnostics"], indent=2))
    print(f"\nMetrics saved: {results_path}")
    print(f"Disagreements saved: {examples_path}")


if __name__ == "__main__":
    main()

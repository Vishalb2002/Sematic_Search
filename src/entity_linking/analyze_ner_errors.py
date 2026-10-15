
from collections import Counter

from datasets import load_dataset
import spacy


DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"
SUPPORTED_LABELS = {"PER", "ORG", "LOC", "MISC"}

# Conservative mapping: do not guess mappings for other spaCy labels.
# LABEL_MAP = {
#     "PERSON": "PER",
#     "ORG": "ORG",
#     "GPE": "LOC",
#     "LOC": "LOC",
# }
LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "PRODUCT": "MISC",
    "NORP": "MISC",
}

MAX_EXAMPLES = 30


def calculate_metrics(true_positive, predicted_total, gold_total):
    precision = (
        true_positive / predicted_total if predicted_total else 0.0
    )
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def main():
    dataset = load_dataset(DATASET_NAME)
    nlp = spacy.load("en_core_web_sm")

    counts = Counter()
    shown = 0

    for doc_index, document in enumerate(dataset["test"], start=1):
        text = document["text"]

        # Gold entities, indexed by exact character span.
        gold = {}

        for entity in document["entities"]:
            label = entity.get("tag")

            if label not in SUPPORTED_LABELS:
                continue

            start = entity["start"]
            end = entity["end"]

            gold[(start, end)] = {
                "label": label,
                "text": text[start:end],
                "qid": entity.get("qid"),
            }

        # spaCy predictions, indexed by exact character span.
        doc = nlp(text)
        predicted = {}

        for entity in doc.ents:
            key = (entity.start_char, entity.end_char)

            predicted[key] = {
                "label": LABEL_MAP.get(entity.label_),
                "original_label": entity.label_,
                "text": entity.text,
            }

        gold_spans = set(gold.keys())
        predicted_spans = set(predicted.keys())

        # Span-only matching ignores entity types.
        exact_spans = gold_spans & predicted_spans
        missed_spans = gold_spans - predicted_spans
        extra_spans = predicted_spans - gold_spans

        counts["gold_spans"] += len(gold_spans)
        counts["predicted_spans"] += len(predicted_spans)
        counts["exact_span_matches"] += len(exact_spans)
        counts["missed_spans"] += len(missed_spans)
        counts["extra_spans"] += len(extra_spans)

        # Exact span AND mapped entity type must match.
        for span in exact_spans:
            gold_label = gold[span]["label"]
            predicted_label = predicted[span]["label"]

            if gold_label == predicted_label:
                counts["correct_span_and_type"] += 1
                counts[f"correct_{gold_label}"] += 1
            else:
                counts["wrong_type_same_span"] += 1

        # Count mapped and unmapped predictions separately.
        for item in predicted.values():
            label = item["label"]

            if label is None:
                counts["unmapped_predictions"] += 1
            else:
                counts[f"predicted_{label}"] += 1

        for item in gold.values():
            counts[f"gold_{item['label']}"] += 1

        # Prepare diagnostic examples.
        errors = []

        for span in sorted(missed_spans):
            errors.append((
                "MISSED GOLD SPAN",
                gold[span]["text"],
                gold[span]["label"],
                "No prediction with the same boundaries",
            ))

        for span in sorted(exact_spans):
            gold_label = gold[span]["label"]
            predicted_label = predicted[span]["label"]

            if gold_label != predicted_label:
                errors.append((
                    "WRONG TYPE / UNMAPPED",
                    gold[span]["text"],
                    gold_label,
                    f"spaCy={predicted[span]['original_label']}, "
                    f"mapped={predicted_label}",
                ))

        for span in sorted(extra_spans):
            item = predicted[span]
            errors.append((
                "EXTRA PREDICTION",
                item["text"],
                "No exact gold span",
                f"spaCy={item['original_label']}, "
                f"mapped={item['label']}",
            ))

        for error in errors:
            counts[error[0]] += 1

            if shown < MAX_EXAMPLES:
                kind, mention, gold_label, prediction = error
                print(f"\n[{kind}]")
                print("Mention:", repr(mention))
                print("Gold:", gold_label)
                print("Prediction:", prediction)
                print("Context:", text[:250].replace("\n", " "))
                shown += 1

        if doc_index % 25 == 0 or doc_index == len(dataset["test"]):
            print(
                f"Processed {doc_index}/{len(dataset['test'])} documents"
            )

    # Aggregate span-only metrics across the complete test set.
    span_precision, span_recall, span_f1 = calculate_metrics(
        counts["exact_span_matches"],
        counts["predicted_spans"],
        counts["gold_spans"],
    )

    # Strict metrics: exact span and correct mapped type.
    strict_precision, strict_recall, strict_f1 = calculate_metrics(
        counts["correct_span_and_type"],
        counts["predicted_spans"],
        counts["gold_spans"],
    )

    print("\n" + "=" * 60)
    print("NER EVALUATION AND ERROR ANALYSIS")
    print("=" * 60)

    print("\nDATASET COUNTS")
    print("Gold entities:", counts["gold_spans"])
    print("Predicted entities:", counts["predicted_spans"])
    print("Exact span matches (regardless of type):",
          counts["exact_span_matches"])
    print("Correct span and mapped type:",
          counts["correct_span_and_type"])
    print("Unmapped spaCy predictions:",
          counts["unmapped_predictions"])

    print("\nSPAN-ONLY EVALUATION")
    print(f"Precision: {span_precision:.4f}")
    print(f"Recall:    {span_recall:.4f}")
    print(f"F1:        {span_f1:.4f}")

    print("\nSTRICT FOUR-CLASS EVALUATION")
    print(f"Precision: {strict_precision:.4f}")
    print(f"Recall:    {strict_recall:.4f}")
    print(f"F1:        {strict_f1:.4f}")

    print("\nPER-TYPE STRICT RESULTS")
    for label in ("PER", "ORG", "LOC", "MISC"):
        precision, recall, f1 = calculate_metrics(
            counts[f"correct_{label}"],
            counts[f"predicted_{label}"],
            counts[f"gold_{label}"],
        )
        print(
            f"{label}: gold={counts[f'gold_{label}']}, "
            f"predicted={counts[f'predicted_{label}']}, "
            f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )

    print("\nERROR COUNTS")
    print("Missed gold spans:", counts["missed_spans"])
    print("Extra predicted spans:", counts["extra_spans"])
    print("Wrong type on exact spans:", counts["wrong_type_same_span"])

    print("\nNOTE")
    print("Unmapped labels count as false positives in strict precision.")
    print("Span-only matching ignores labels but still requires exact offsets.")
    print("The label mapping is provisional and must be validated.")
    print("Boundary mismatches may appear as both a miss and an extra span.")


if __name__ == "__main__":
    main()

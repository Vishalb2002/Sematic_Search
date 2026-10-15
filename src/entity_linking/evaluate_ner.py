
from collections import Counter
from datasets import load_dataset
import spacy

DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"

# Initial label mapping. Review these mappings before reporting results.
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

def main():
    dataset = load_dataset(DATASET_NAME)
    nlp = spacy.load("en_core_web_sm")

    # Exact span + exact type matching.
    gold_total = 0
    predicted_total = 0
    true_positive = 0
    gold_by_type = Counter()
    predicted_by_type = Counter()
    correct_by_type = Counter()

    for doc_index, example in enumerate(dataset["test"]):
        text = example["text"]

        gold = set()
        for entity in example["entities"]:
            start = entity["start"]
            end = entity["end"]
            label = entity.get("tag")

            if label in {"PER", "ORG", "LOC", "MISC"}:
                gold.add((start, end, label))

        predicted = set()
        doc = nlp(text)

        for entity in doc.ents:
            mapped_label = LABEL_MAP.get(entity.label_)
            
            predicted.add(
                (entity.start_char, entity.end_char, mapped_label)
            )

        gold_total += len(gold)
        predicted_total += len(predicted)

        for _, _, label in gold:
            gold_by_type[label] += 1

        for _, _, label in predicted:
            predicted_by_type[label] += 1

        matches = gold & predicted
        true_positive += len(matches)

        for _, _, label in matches:
            correct_by_type[label] += 1

        if (doc_index + 1) % 25 == 0:
            print(
                f"Processed {doc_index + 1}/"
                f"{len(dataset['test'])} documents"
            )

    precision = (
        true_positive / predicted_total if predicted_total else 0.0
    )
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print("\n" + "=" * 55)
    print("SPA Cy NER EVALUATION — EXACT SPAN + TYPE")
    print("=" * 55)
    print(f"Gold entities:      {gold_total}")
    print(f"Predicted entities: {predicted_total}")
    print(f"Exact true positives: {true_positive}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nPer-type results:")
    for label in ["PER", "ORG", "LOC", "MISC"]:
        tp = correct_by_type[label]
        pred = predicted_by_type[label]
        gold_count = gold_by_type[label]

        p = tp / pred if pred else 0.0
        r = tp / gold_count if gold_count else 0.0
        type_f1 = 2 * p * r / (p + r) if p + r else 0.0

        print(
            f"{label}: gold={gold_count}, predicted={pred}, "
            f"TP={tp}, P={p:.4f}, R={r:.4f}, F1={type_f1:.4f}"
        )

    print("\nImportant: this is an initial baseline.")
    print("Unmapped spaCy predictions count as false positives.")
    print("Label mappings still require validation.")

if __name__ == "__main__":
    main()

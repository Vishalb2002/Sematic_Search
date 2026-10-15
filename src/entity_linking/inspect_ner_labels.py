
from collections import Counter
from datasets import load_dataset
import spacy

DATASET_NAME = "cyanic-selkie/aida-conll-yago-wikidata"

dataset = load_dataset(DATASET_NAME)
nlp = spacy.load("en_core_web_sm")

gold_labels = Counter()
gold_examples = {}

for document in dataset["test"]:
    for entity in document["entities"]:
        label = entity.get("tag")
        if label is not None:
            gold_labels[str(label)] += 1
            gold_examples.setdefault(str(label), entity.get("title"))

print("\nDataset entity labels:")
for label, count in gold_labels.most_common():
    print(f"{label}: {count} | Example: {gold_examples[label]}")

predicted_labels = Counter()

for document in list(dataset["test"].select(range(20))):
    doc = nlp(document["text"])
    for entity in doc.ents:
        predicted_labels[entity.label_] += 1

print("\nspaCy entity labels (first 20 test documents):")
for label, count in predicted_labels.most_common():
    print(f"{label}: {count}")

print("\nExample gold annotation:")
print(dataset["test"][0]["entities"][:3])

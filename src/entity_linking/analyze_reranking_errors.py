import json
from collections import Counter
from pathlib import Path

INPUT = Path(
    "outputs/entity_linking_validation/validation_predictions.json"
)

with INPUT.open(encoding="utf-8") as f:
    examples = json.load(f)

fixed = []
introduced = []
both_wrong = []
both_correct = []

for ex in examples:
    gold = str(ex["gold_id"])
    before = ex["retrieved"]
    after = ex["reranked"]

    before_id = str(before[0]["entity_id"]) if before else None
    after_id = str(after[0]["entity_id"]) if after else None

    before_ok = before_id == gold
    after_ok = after_id == gold

    if not before_ok and after_ok:
        fixed.append(ex)
    elif before_ok and not after_ok:
        introduced.append(ex)
    elif before_ok and after_ok:
        both_correct.append(ex)
    else:
        both_wrong.append(ex)


def show_cases(title, cases, limit=12):
    print(f"\n{'=' * 80}\n{title}: {len(cases)} cases")
    for ex in cases[:limit]:
        before = ex["retrieved"]
        after = ex["reranked"]

        print("\nMention:", ex["mention"])
        print("Context:", ex["context"][:300])
        print("Gold:", ex["gold_title"], "|", ex["gold_id"])

        if before:
            print(
                "SBERT top-1:",
                before[0]["name"],
                "|",
                before[0]["entity_id"],
                "| distance:",
                round(before[0]["distance"], 4),
            )

        if after:
            print(
                "Reranked top-1:",
                after[0]["name"],
                "|",
                after[0]["entity_id"],
                "| score:",
                round(after[0]["rerank_score"], 4),
            )

        print("Top reranked candidates:")
        for candidate in after[:5]:
            print(
                " ",
                candidate["name"],
                "|",
                candidate["entity_id"],
                "| score:",
                round(candidate["rerank_score"], 4),
                "| gold:",
                str(candidate["entity_id"]) == str(ex["gold_id"]),
            )


print("Total examples:", len(examples))
print("Fixed:", len(fixed))
print("Introduced:", len(introduced))
print("Both correct:", len(both_correct))
print("Both wrong:", len(both_wrong))

print("\nEntity types for introduced errors:")
print(Counter(
    str(ex.get("entity_type"))
    for ex in introduced
))

print("\nEntity types for fixed errors:")
print(Counter(
    str(ex.get("entity_type"))
    for ex in fixed
))

show_cases("RERANKING INTRODUCED ERRORS", introduced)
show_cases("RERANKING FIXED ERRORS", fixed)

# Save all introduced errors for deeper analysis.
output = Path(
    "outputs/entity_linking_validation/introduced_errors.json"
)
with output.open("w", encoding="utf-8") as f:
    json.dump(introduced, f, indent=2, ensure_ascii=False)

print("\nSaved introduced errors to:", output)

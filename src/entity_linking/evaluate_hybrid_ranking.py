
import json
from pathlib import Path

INPUT = Path(
    "outputs/entity_linking_validation/validation_predictions.json"
)

with INPUT.open(encoding="utf-8") as f:
    examples = json.load(f)


def get_ranked_ids(candidates):
    return [str(c["entity_id"]) for c in candidates]


def evaluate(rankings, gold_ids):
    total = len(gold_ids)
    result = {}

    for k in (1, 5, 10):
        hits = sum(
            gold in ranking[:k]
            for ranking, gold in zip(rankings, gold_ids)
        )
        result[f"recall@{k}"] = hits / total if total else 0

    reciprocal_ranks = []
    top1 = 0

    for ranking, gold in zip(rankings, gold_ids):
        if ranking and ranking[0] == gold:
            top1 += 1

        try:
            reciprocal_ranks.append(1 / (ranking.index(gold) + 1))
        except ValueError:
            reciprocal_ranks.append(0.0)

    result["mrr"] = (
        sum(reciprocal_ranks) / total if total else 0
    )
    result["top1_accuracy"] = top1 / total if total else 0

    return result


def reciprocal_rank_fusion(original, reranked, alpha, k=10):
    """
    alpha controls the weight of the original SBERT ranking.
    1-alpha controls the CrossEncoder ranking.
    """
    scores = {}

    for rank, entity_id in enumerate(original, start=1):
        scores[entity_id] = (
            scores.get(entity_id, 0.0)
            + alpha / (k + rank)
        )

    for rank, entity_id in enumerate(reranked, start=1):
        scores[entity_id] = (
            scores.get(entity_id, 0.0)
            + (1 - alpha) / (k + rank)
        )

    return sorted(
        scores,
        key=lambda entity_id: scores[entity_id],
        reverse=True,
    )


gold_ids = [str(ex["gold_id"]) for ex in examples]
original_rankings = [
    get_ranked_ids(ex["retrieved"]) for ex in examples
]
reranked_rankings = [
    get_ranked_ids(ex["reranked"]) for ex in examples
]

covered_indices = [
    i for i, ranking in enumerate(original_rankings)
    if gold_ids[i] in ranking
]

print("Examples:", len(examples))
print("\nSBERT baseline:")
print(json.dumps(
    evaluate(original_rankings, gold_ids), indent=2
))

print("\nCrossEncoder baseline:")
print(json.dumps(
    evaluate(reranked_rankings, gold_ids), indent=2
))

print("\nHybrid RRF experiments:")
results = []

for alpha in (0.25, 0.50, 0.75):
    hybrid_rankings = [
        reciprocal_rank_fusion(
            original, reranked, alpha=alpha
        )
        for original, reranked
        in zip(original_rankings, reranked_rankings)
    ]

    all_metrics = evaluate(hybrid_rankings, gold_ids)

    covered_rankings = [
        hybrid_rankings[i] for i in covered_indices
    ]
    covered_gold = [gold_ids[i] for i in covered_indices]

    covered_metrics = evaluate(covered_rankings, covered_gold)

    row = {
        "alpha_sbert": alpha,
        "alpha_cross_encoder": 1 - alpha,
        "all_mentions": all_metrics,
        "covered_mentions": covered_metrics,
    }
    results.append(row)

    print(f"\nSBERT weight={alpha:.2f}; "
          f"CrossEncoder weight={1-alpha:.2f}")
    print("All mentions:", json.dumps(all_metrics))
    print("Covered mentions:", json.dumps(covered_metrics))

output = Path(
    "outputs/entity_linking_validation/hybrid_rrf_results.json"
)
with output.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", output)


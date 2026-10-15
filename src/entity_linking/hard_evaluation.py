
from typing import Dict, List

from src.entity_linking.candidate_retriever import CandidateRetriever
from src.entity_linking.evaluation import entities


# Harder contextual cases using the existing entity catalogue.
HARD_TEST_DATA = [
    {
        "mention": "Apple",
        "context": (
            "The technology firm announced its latest iPhone, "
            "updated operating systems, and new computer hardware."
        ),
        "gold_entity_id": "ORG_APPLE",
    },
    {
        "mention": "apple",
        "context": (
            "The farmer harvested ripe fruit from the orchard "
            "and used the apples to prepare a pie."
        ),
        "gold_entity_id": "CONCEPT_APPLE_FRUIT",
    },
    {
        "mention": "Mercury",
        "context": (
            "Astronomers studied the innermost planet while "
            "observing its orbit around the Sun."
        ),
        "gold_entity_id": "PLANET_MERCURY",
    },
    {
        "mention": "Mercury",
        "context": (
            "The laboratory identified the toxic metal by its "
            "chemical symbol Hg in the contaminated sample."
        ),
        "gold_entity_id": "ELEMENT_MERCURY",
    },
    {
        "mention": "Python",
        "context": (
            "The developer implemented a machine learning pipeline "
            "using libraries, functions, and data-processing code."
        ),
        "gold_entity_id": "LANG_PYTHON",
    },
    {
        "mention": "python",
        "context": (
            "The reptile coiled around a branch before constricting "
            "its prey in the forest."
        ),
        "gold_entity_id": "ANIMAL_PYTHON",
    },
    {
        "mention": "Amazon",
        "context": (
            "The retailer expanded its cloud infrastructure, "
            "online shopping services, and computing platform."
        ),
        "gold_entity_id": "COMP_AMAZON",
    },
    {
        "mention": "Amazon",
        "context": (
            "The explorers travelled along the South American "
            "waterway surrounded by rainforest."
        ),
        "gold_entity_id": "RIVER_AMAZON",
    },
    {
        "mention": "Jordan",
        "context": (
            "The basketball player became famous for scoring, "
            "championships, and his professional sports career."
        ),
        "gold_entity_id": "PERSON_JORDAN",
    },
    {
        "mention": "Jordan",
        "context": (
            "The Middle Eastern nation has Amman as its capital "
            "and shares borders with neighbouring countries."
        ),
        "gold_entity_id": "COUNTRY_JORDAN",
    },
    {
        "mention": "Java",
        "context": (
            "The software engineer used classes, objects, "
            "and a virtual machine to build an application."
        ),
        "gold_entity_id": "LANG_JAVA",
    },
    {
        "mention": "Google",
        "context": (
            "The technology company develops web search products, "
            "online services, and artificial intelligence systems."
        ),
        "gold_entity_id": "ORG_GOOGLE",
    },
]


# These mentions do not have corresponding entities in our catalogue.
NIL_TEST_DATA = [
    {
        "mention": "Zorvax",
        "context": "Zorvax is a fictional company in this test example.",
        "expected_nil": True,
    },
    {
        "mention": "Qeltrium",
        "context": "Qeltrium is an invented entity used for evaluation.",
        "expected_nil": True,
    },
    {
        "mention": "Apple",
        "context": "Apple released a new iPhone.",
        "expected_nil": False,
        "gold_entity_id": "ORG_APPLE",
    },
    {
        "mention": "Mercury",
        "context": "Mercury is the closest planet to the Sun.",
        "expected_nil": False,
        "gold_entity_id": "PLANET_MERCURY",
    },
]


def reciprocal_rank(candidates: List[Dict], gold_id: str) -> float:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["entity_id"] == gold_id:
            return 1.0 / rank
    return 0.0


def evaluate_linking(retriever: CandidateRetriever) -> None:
    baseline_ranks = []
    reranked_ranks = []
    candidate_recall = []

    print("\nHARD ENTITY-LINKING EVALUATION")
    print("=" * 60)

    for example in HARD_TEST_DATA:
        mention = example["mention"]
        context = example["context"]
        gold_id = example["gold_entity_id"]

        candidates = retriever.search(
            mention=mention,
            context=context,
            top_k=5,
        )

        baseline_ids = [
            item["entity_id"] for item in candidates
        ]

        candidate_recall.append(
            float(gold_id in baseline_ids)
        )
        baseline_ranks.append(
            reciprocal_rank(candidates, gold_id)
        )

        reranked = retriever.rerank_candidates(
            mention=mention,
            context=context,
            candidates=candidates,
            top_k=5,
        ) if candidates else []

        reranked_ids = [
            item["entity_id"] for item in reranked
        ]
        reranked_ranks.append(
            reciprocal_rank(reranked, gold_id)
        )

        baseline_rank = (
            baseline_ids.index(gold_id) + 1
            if gold_id in baseline_ids else "Missing"
        )
        reranked_rank = (
            reranked_ids.index(gold_id) + 1
            if gold_id in reranked_ids else "Missing"
        )

        print(
            f"\nMention: {mention!r}"
            f"\nGold: {gold_id}"
            f"\nSBERT rank: {baseline_rank}"
            f"\nReranked rank: {reranked_rank}"
        )

    n = len(HARD_TEST_DATA)
    print("\nRanking metrics")
    print("-" * 40)
    print(f"Examples: {n}")
    print(f"Candidate Recall@5: {sum(candidate_recall) / n:.3f}")
    print(f"SBERT MRR: {sum(baseline_ranks) / n:.3f}")
    print(f"SBERT + CrossEncoder MRR: {sum(reranked_ranks) / n:.3f}")
    print(
        "MRR change: "
        f"{(sum(reranked_ranks) - sum(baseline_ranks)) / n:+.3f}"
    )


def evaluate_nil(retriever: CandidateRetriever) -> None:
    # This is a provisional threshold, not a calibrated value.
    threshold = 0.0

    correct = 0
    true_nil = 0
    predicted_nil = 0
    true_positive_nil = 0

    print("\nNIL DETECTION EVALUATION")
    print("=" * 60)
    print(f"Provisional CrossEncoder threshold: {threshold}")

    for example in NIL_TEST_DATA:
        result = retriever.predict_entity(
            mention=example["mention"],
            context=example["context"],
            top_k=5,
            min_score=threshold,
        )

        expected_nil = example["expected_nil"]
        predicted_is_nil = result["decision"] == "NIL"

        correct += int(predicted_is_nil == expected_nil)
        true_nil += int(expected_nil)
        predicted_nil += int(predicted_is_nil)

        if expected_nil and predicted_is_nil:
            true_positive_nil += 1

        print(
            f"\nMention: {example['mention']!r}"
            f"\nExpected: {'NIL' if expected_nil else 'LINKED'}"
            f"\nPredicted: {result['decision']}"
            f"\nEntity: {result['entity_name']}"
            f"\nScore: {result['score']}"
        )

    n = len(NIL_TEST_DATA)
    precision = (
        true_positive_nil / predicted_nil
        if predicted_nil else 0.0
    )
    recall = (
        true_positive_nil / true_nil
        if true_nil else 0.0
    )

    print("\nNIL metrics")
    print("-" * 40)
    print(f"Examples: {n}")
    print(f"Decision accuracy: {correct / n:.3f}")
    print(f"NIL precision: {precision:.3f}")
    print(f"NIL recall: {recall:.3f}")


def main() -> None:
    retriever = CandidateRetriever()

    # Ensure the catalogue is populated with the known entities.
    retriever.add_entities(entities)

    print(f"Entity catalogue count: {retriever.collection.count()}")

    evaluate_linking(retriever)
    evaluate_nil(retriever)


if __name__ == "__main__":
    main()

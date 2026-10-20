
import re
import unicodedata
from collections import defaultdict

from src.entity_linking.candidate_retriever import CandidateRetriever


class HybridCandidateRetriever:
    """
    Combines:
    1. Exact canonical-name matching
    2. Exact alias matching
    3. Existing SBERT semantic retrieval

    Does not modify the existing ChromaDB catalogue.
    """

    def __init__(self):
        self.retriever = CandidateRetriever()
        self.alias_index = defaultdict(set)
        self.entity_data = {}

        self._build_lexical_index()

    @staticmethod
    def normalize(text):
        """Normalize case, Unicode, and punctuation for matching."""
        text = unicodedata.normalize("NFKC", str(text))
        text = text.casefold()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def _build_lexical_index(self):
        stored = self.retriever.collection.get(
            include=["metadatas"]
        )

        for entity_id, metadata in zip(
            stored["ids"], stored["metadatas"]
        ):
            entity_id = str(entity_id)
            name = metadata.get("name", "")
            aliases_raw = metadata.get("aliases", "")

            aliases = [
                alias.strip()
                for alias in aliases_raw.split("|")
                if alias.strip()
            ]

            self.entity_data[entity_id] = {
                "name": name,
                "entity_type": metadata.get(
                    "entity_type", "UNKNOWN"
                ),
                "description": metadata.get("description", ""),
                "aliases": aliases,
            }

            # Index canonical name and every alias.
            for value in [name, *aliases]:
                normalized = self.normalize(value)

                if normalized:
                    self.alias_index[normalized].add(entity_id)

        print("Lexical entities indexed:", len(self.entity_data))
        print("Unique name/alias keys:", len(self.alias_index))

    def search(
        self,
        mention,
        context="",
        top_k=10,
        entity_type=None,
        semantic_top_k=30,
    ):
        """
        Return a merged candidate list.

        Exact canonical-name matches rank first, followed by exact
        alias matches, then SBERT candidates. Exact matches can be
        ambiguous, so all matching entities are retained.
        """
        normalized_mention = self.normalize(mention)
        candidates = {}
        lexical_matches = set()

        # Exact normalized name or alias matches.
        lexical_matches.update(
            self.alias_index.get(normalized_mention, set())
        )

        # Retrieve semantic candidates using the existing retriever.
        semantic_candidates = self.retriever.search(
            mention=mention,
            context=context,
            top_k=semantic_top_k,
            entity_type=entity_type,
        )

        semantic_rank = {}

        for rank, candidate in enumerate(
            semantic_candidates, start=1
        ):
            entity_id = str(candidate["entity_id"])
            semantic_rank[entity_id] = rank
            candidates[entity_id] = dict(candidate)

        # Add exact lexical matches even if SBERT did not retrieve them.
        for entity_id in lexical_matches:
            data = self.entity_data[entity_id]

            if (
                entity_type
                and data["entity_type"] != entity_type
            ):
                continue

            if entity_id not in candidates:
                candidates[entity_id] = {
                    "entity_id": entity_id,
                    "name": data["name"],
                    "entity_type": data["entity_type"],
                    "description": data["description"],
                    "aliases": data["aliases"],
                    "distance": None,
                }

        def ranking_key(item):
            entity_id, candidate = item

            normalized_name = self.normalize(
                candidate.get("name", "")
            )

            # Prioritize exact canonical name, then exact alias,
            # then semantic ranking.
            if normalized_name == normalized_mention:
                lexical_priority = 0
            elif entity_id in lexical_matches:
                lexical_priority = 1
            else:
                lexical_priority = 2

            return (
                lexical_priority,
                semantic_rank.get(entity_id, float("inf")),
            )

        ranked = sorted(
            candidates.items(),
            key=ranking_key,
        )

        results = []

        for entity_id, candidate in ranked[:top_k]:
            candidate["lexical_match"] = (
                entity_id in lexical_matches
            )
            candidate["semantic_rank"] = semantic_rank.get(
                entity_id
            )
            results.append(candidate)

        return results

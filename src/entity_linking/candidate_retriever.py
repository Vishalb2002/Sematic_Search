
from typing import Dict, List, Optional
import chromadb

from src.embeddings.embedding_model import EmbeddingModel


class CandidateRetriever:
    """
    Retrieves canonical entity candidates using SBERT
    and a dedicated ChromaDB collection.

    This collection is separate from the research-paper
    collection used by the existing semantic-search system.
    """

    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "entity_catalogue",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = EmbeddingModel.get_model(model_name)

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Canonical entity catalogue"},
        )

    def add_entities(self, entities: List[Dict]) -> None:
        """
        Add entity records.

        Each record requires:
            entity_id: stable canonical ID
            name: canonical entity name
            description: entity description

        Optional:
            entity_type: PERSON, ORG, CONCEPT, etc.
            aliases: list of alternative names
        """
        if not entities:
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for entity in entities:
            entity_id = str(entity["entity_id"])
            name = entity["name"].strip()
            description = entity["description"].strip()

            aliases = entity.get("aliases", [])
            entity_type = entity.get("entity_type", "UNKNOWN")

            searchable_text = (
                f"Name: {name}. "
                f"Aliases: {', '.join(aliases)}. "
                f"Type: {entity_type}. "
                f"Description: {description}"
            )

            ids.append(entity_id)
            documents.append(searchable_text)
            embeddings.append(
                self.model.encode(
                    searchable_text,
                    convert_to_numpy=True,
                ).tolist()
            )

            metadatas.append({
                "entity_id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "description": description,
                "aliases": " | ".join(aliases),
            })

        # Upsert allows the same entity ID to be updated safely.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        mention: str,
        context: str = "",
        top_k: int = 5,
        entity_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve possible canonical entities for a mention.

        ChromaDB returns distances; smaller distances generally
        indicate closer vectors for the configured metric.
        A distance is not a calibrated probability.
        """
        if not mention or not mention.strip():
            return []

        if top_k < 1 or self.collection.count() == 0:
            return []

        query_text = (
            f"Mention: {mention.strip()}. "
            f"Context: {context.strip()}"
        )

        query_embedding = self.model.encode(
            query_text,
            convert_to_numpy=True,
        ).tolist()

        # Retrieve broadly, then apply an optional type filter.
        candidate_count = min(
            self.collection.count(),
            max(top_k * 5, top_k),
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            candidate_type = metadata.get("entity_type", "UNKNOWN")

            if (
                entity_type
                and candidate_type != entity_type
                and candidate_type != "UNKNOWN"
            ):
                continue

            candidates.append({
                "entity_id": metadata["entity_id"],
                "name": metadata["name"],
                "entity_type": candidate_type,
                "description": metadata["description"],
                "aliases": metadata.get("aliases", ""),
                "distance": float(distance),
                "retrieval_text": document,
            })

        return candidates[:top_k]
    

    def rerank_candidates(
        self,
        mention: str,
        context: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank retrieved entity candidates using the existing
        CrossEncoder model and mention context.
        """
        if not candidates:
            return []

        from src.retrieval.cross_encoder import CrossEncoderModel

        cross_encoder = CrossEncoderModel.get_model()

        query = (
            f"Entity mention: {mention}\n"
            f"Context: {context}"
        )

        candidate_pairs = [
            (
                query,
                (
                    f"Entity: {candidate['name']}. "
                    f"Type: {candidate['entity_type']}. "
                    f"Description: {candidate['description']}. "
                    f"Aliases: {candidate.get('aliases', '')}"
                ),
            )
            for candidate in candidates
        ]

        scores = cross_encoder.predict(candidate_pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]



    def predict_entity(
        self,
        mention: str,
        context: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> Dict:
        """
        Return the best candidate or NIL if the best
        CrossEncoder score is below the chosen threshold.

        min_score is a tunable decision threshold, not
        a calibrated probability.
        """
        candidates = self.search(
            mention=mention,
            context=context,
            top_k=top_k,
        )

        if not candidates:
            return {
                "mention": mention,
                "entity_id": None,
                "entity_name": None,
                "decision": "NIL",
                "score": None,
            }

        reranked = self.rerank_candidates(
            mention=mention,
            context=context,
            candidates=candidates,
            top_k=top_k,
        )

        best = reranked[0]
        score = best["rerank_score"]

        if score < min_score:
            return {
                "mention": mention,
                "entity_id": None,
                "entity_name": None,
                "decision": "NIL",
                "score": score,
            }

        return {
            "mention": mention,
            "entity_id": best["entity_id"],
            "entity_name": best["name"],
            "decision": "LINKED",
            "score": score,
        }


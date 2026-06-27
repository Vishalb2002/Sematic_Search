from retrieval.cross_encoder import CrossEncoderModel


class ReRanker:
    """
    Re-ranks the retrieved document chunks using
    a CrossEncoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading CrossEncoder...")

        self.model = CrossEncoderModel.get_model(model_name)

        print("CrossEncoder Ready.\n")

    def rerank(
        self,
        query: str,
        retrieved_chunks: list,
        top_k: int = 5
    ):
        """
        Re-rank the retrieved chunks.

        Parameters
        ----------
        query : str
            User query.

        retrieved_chunks : list
            Output from Retriever.

        top_k : int
            Number of final chunks to return.
        """

        # Create (query, chunk) pairs
        sentence_pairs = [
            (query, chunk["text"])
            for chunk in retrieved_chunks
        ]

        # Predict relevance scores
        scores = self.model.predict(sentence_pairs)

        # Attach scores to chunks
        for chunk, score in zip(retrieved_chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by highest score
        reranked_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # Keep only Top-K
        reranked_chunks = reranked_chunks[:top_k]

        # Update ranks
        for index, chunk in enumerate(reranked_chunks, start=1):
            chunk["rank"] = index

        return reranked_chunks
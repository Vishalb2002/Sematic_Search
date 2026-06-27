import chromadb
from embeddings.embedding_model import EmbeddingModel


class Retriever:
    """
    Retrieves the most relevant document chunks
    from ChromaDB using semantic search.
    """

    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "gesture_research_papers",
        model_name: str = "all-MiniLM-L6-v2"
    ):

        print("Loading Retriever...")

        # Load SBERT Model
        self.model = EmbeddingModel.get_model(model_name)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        print("Retriever Ready.\n")

    # def search(
    #     self,
    #     query: str,
    #     top_k: int = 5
    #     ):

    #     query_embedding = self.model.encode(
    #     query,
    #     convert_to_numpy=True
    #      ).tolist()

    #     results = self.collection.query(
    #     query_embeddings=[query_embedding],
    #     n_results=top_k
    #     )

    #     formatted_results = []

    #     documents = results["documents"][0]
    #     metadatas = results["metadatas"][0]
    #     distances = results["distances"][0]

    #     for doc, meta, distance in zip(
    #     documents,
    #     metadatas,
    #     distances
    #     ):

    #         formatted_results.append({
    #         "source": meta["source"],
    #         "chunk_id": meta["chunk_id"],
    #         "distance": distance,
    #         "text": doc
    #         })

    #     return formatted_results
    def search(
    self,
    query: str,
    top_k: int = 20
    ):

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        formatted_results = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1
        ):

            formatted_results.append({
                "rank": rank,
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "distance": distance,
                "text": doc
            })

        return formatted_results
    
    def build_context(self, retrieved_chunks):
        """
        Combine retrieved chunks into one context
        for the LLM.
        """

        context = ""

        for chunk in retrieved_chunks:

            context += (
                f"Rank: {chunk['rank']}\n"
                f"Source: {chunk['source']}\n"
                f"Chunk: {chunk['chunk_id']}\n\n"
                f"{chunk['text']}\n"
                "\n"
                "----------------------------------------\n\n"
            )

        return context
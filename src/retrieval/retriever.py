import chromadb
from sentence_transformers import SentenceTransformer


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
        self.model = SentenceTransformer(model_name)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        print("Retriever Ready.\n")

    def search(
    self,
    query: str,
    top_k: int = 5
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

        for doc, meta, distance in zip(
        documents,
        metadatas,
        distances
        ):

            formatted_results.append({
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "distance": distance,
            "text": doc
            })

        return formatted_results
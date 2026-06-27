from typing import List, Dict
import chromadb


class VectorStore:
    """
    Handles storing and retrieving embeddings
    using ChromaDB.
    """

    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "gesture_research_papers"
    ):

        print("Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print("ChromaDB Ready.\n")

    def add_embeddings(
        self,
        embedded_chunks: List[Dict]
    ):
        """
        Store embeddings into ChromaDB.
        """

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in embedded_chunks:

            ids.append(
                f"{chunk['source']}_{chunk['chunk_id']}"
            )

            embeddings.append(
                chunk["embedding"]
            )

            documents.append(
                chunk["text"]
            )

            metadatas.append({
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"]
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Stored {len(ids)} embeddings successfully.")

    def total_chunks(self):

        return self.collection.count()
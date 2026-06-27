from retrieval.retriever import Retriever
from retrieval.reranker import ReRanker


def main():

    query = "Explain Vision Transformer"

    # -------------------------
    # Step 1 : Retrieve
    # -------------------------
    retriever = Retriever()

    retrieved_chunks = retriever.search(
        query=query,
        top_k=20
    )

    print(f"\nRetrieved {len(retrieved_chunks)} chunks.\n")

    # -------------------------
    # Step 2 : Re-rank
    # -------------------------
    reranker = ReRanker()

    final_chunks = reranker.rerank(
        query=query,
        retrieved_chunks=retrieved_chunks,
        top_k=5
    )

    print("\nAfter Re-ranking\n")

    for chunk in final_chunks:

        print("=" * 100)
        print(f"Rank          : {chunk['rank']}")
        print(f"Source        : {chunk['source']}")
        print(f"Chunk ID      : {chunk['chunk_id']}")
        print(f"Distance      : {chunk['distance']:.4f}")
        print(f"Rerank Score  : {chunk['rerank_score']:.4f}")
        print()
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()
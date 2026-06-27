from retrieval.retriever import Retriever
from retrieval.reranker import ReRanker
from llm.prompt_builder import PromptBuilder
from llm.llm import LLM


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

    # for chunk in final_chunks:

    #     print("=" * 100)
    #     print(f"Rank          : {chunk['rank']}")
    #     print(f"Source        : {chunk['source']}")
    #     print(f"Chunk ID      : {chunk['chunk_id']}")
    #     print(f"Distance      : {chunk['distance']:.4f}")
    #     print(f"Rerank Score  : {chunk['rerank_score']:.4f}")
    #     print()
    #     print(chunk["text"])
    #     print()

    builder = PromptBuilder()

    prompt = builder.build_prompt(
        query=query,
        retrieved_chunks=final_chunks
    )

    print(prompt)

    llm = LLM()

    answer = llm.generate_answer(prompt)

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(answer)


if __name__ == "__main__":
    main()
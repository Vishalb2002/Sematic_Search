# from retrieval.retriever import Retriever
# from retrieval.reranker import ReRanker
# from llm.prompt_builder import PromptBuilder
# from llm.llm import LLM


# def main():

#     query = "Explain Vision Transformer"

#     # -------------------------
#     # Step 1 : Retrieve
#     # -------------------------
#     retriever = Retriever()

#     retrieved_chunks = retriever.search(
#         query=query,
#         top_k=20
#     )

#     print(f"\nRetrieved {len(retrieved_chunks)} chunks.\n")

#     # -------------------------
#     # Step 2 : Re-rank
#     # -------------------------
#     reranker = ReRanker()

#     final_chunks = reranker.rerank(
#         query=query,
#         retrieved_chunks=retrieved_chunks,
#         top_k=5
#     )

#     print("\nAfter Re-ranking\n")

#     # for chunk in final_chunks:

#     #     print("=" * 100)
#     #     print(f"Rank          : {chunk['rank']}")
#     #     print(f"Source        : {chunk['source']}")
#     #     print(f"Chunk ID      : {chunk['chunk_id']}")
#     #     print(f"Distance      : {chunk['distance']:.4f}")
#     #     print(f"Rerank Score  : {chunk['rerank_score']:.4f}")
#     #     print()
#     #     print(chunk["text"])
#     #     print()

#     builder = PromptBuilder()

#     prompt = builder.build_prompt(
#         query=query,
#         retrieved_chunks=final_chunks
#     )

#     print(prompt)

#     llm = LLM()

#     answer = llm.generate_answer(prompt)

#     print("\n" + "=" * 100)
#     print("FINAL ANSWER")
#     print("=" * 100)
#     print(answer)


# if __name__ == "__main__":
#     main()
from retrieval.retriever import Retriever
from retrieval.reranker import ReRanker
from llm.prompt_builder import PromptBuilder
from llm.llm import LLM


def main():

    print("=" * 70)
    print(" Semantic Search & Intelligent Q&A System ")
    print("=" * 70)
    print("Type 'exit' to quit.\n")

    # Load models only once
    retriever = Retriever()
    reranker = ReRanker()
    builder = PromptBuilder()
    llm = LLM()

    while True:

        query = input("\nAsk your question: ").strip()

        # Exit
        if query.lower() == "exit":
            print("\nThank you for using the system!")
            break

        # Ignore empty input
        if not query:
            print("Please enter a valid question.")
            continue

        # -------------------------
        # Step 1 : Retrieve
        # -------------------------

        retrieved_chunks = retriever.search(
            query=query,
            top_k=20
        )

        # -------------------------
        # Step 2 : Re-rank
        # -------------------------

        final_chunks = reranker.rerank(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=5
        )

        # -------------------------
        # Step 3 : Prompt Builder
        # -------------------------

        prompt = builder.build_prompt(
            query=query,
            retrieved_chunks=final_chunks
        )

        # -------------------------
        # Step 4 : LLM
        # -------------------------

        answer = llm.generate_answer(prompt)

        print("\n" + "=" * 70)
        print("Answer\n")
        print(answer)
        print("=" * 70)


if __name__ == "__main__":
    main()
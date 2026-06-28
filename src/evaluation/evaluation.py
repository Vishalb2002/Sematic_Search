from retrieval.retriever import Retriever
from retrieval.reranker import ReRanker


class Evaluator:

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = ReRanker()

        self.test_queries = [
            "What is Vision Transformer?",
            "How does CNN differ from Vision Transformer?",
            "What is hand gesture recognition?",
            "Explain self-attention mechanism.",
            "What datasets are used for gesture recognition?",
            "Explain transformer encoder.",
            "What are the advantages of Vision Transformer?",
            "How is image classification performed?",
            "What preprocessing techniques are used?",
            "Which papers discuss CNN architectures?"
        ]

    def evaluate(self):

        print("=" * 80)
        print("Semantic Search Evaluation")
        print("=" * 80)

        for index, query in enumerate(self.test_queries, start=1):

            print(f"\nQuery {index}: {query}")

            retrieved = self.retriever.search(
                query=query,
                top_k=20
            )

            reranked = self.reranker.rerank(
                query=query,
                retrieved_chunks=retrieved,
                top_k=5
            )

            print("\nTop Retrieved Sources:")

            for chunk in reranked:

                print(
                    f"  Rank {chunk['rank']} "
                    f"| Source: {chunk['source']} "
                    f"| Score: {chunk['rerank_score']:.2f}"
                )

        print("\nEvaluation Complete.")
        print("Inspect the retrieved sources manually to perform qualitative analysis.")


if __name__ == "__main__":

    evaluator = Evaluator()
    evaluator.evaluate()
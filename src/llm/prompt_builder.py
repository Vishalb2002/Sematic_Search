class PromptBuilder:
    """
    Builds prompts for the LLM using the
    retrieved document chunks.
    """

    def __init__(self):
        pass

    def build_context(self, retrieved_chunks: list) -> str:
        """
        Convert retrieved chunks into a readable context.
        """

        context = []

        for chunk in retrieved_chunks:

            context.append(
                f"""
Source   : {chunk['source']}
Chunk ID : {chunk['chunk_id']}

Content:
{chunk['text']}
"""
            )

        return "\n" + ("-" * 80 + "\n").join(context)

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list
    ) -> str:
        """
        Create the final prompt that will be
        sent to the LLM.
        """

        context = self.build_context(retrieved_chunks)

        prompt = f"""
You are an AI Research Assistant.

Answer the user's question ONLY using the information
provided in the context.

Rules:

1. Do not make up information.
2. If the answer is not available in the context,
   reply with:

"I couldn't find enough information in the provided research papers."

3. If multiple papers discuss the topic,
   summarize them together.

4. Mention the source paper whenever possible.

======================
CONTEXT
======================

{context}

======================
QUESTION
======================

{query}

======================
ANSWER
======================
"""

        return prompt
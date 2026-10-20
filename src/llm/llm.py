# import ollama


# class LLM:
#     """
#     Handles communication with the Ollama LLM.
#     """

#     def __init__(
#         self,
#         model_name: str = "llama3.2:3b"
#     ):
#         self.model_name = model_name

#     def generate_answer(
#         self,
#         prompt: str
#     ) -> str:
#         """
#         Send prompt to Ollama and return the response.
#         """

#         response = ollama.chat(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         return response["message"]["content"]


import ollama


class LLM:
    """Communicate with an Ollama model."""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name

    def generate_answer(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 128,
            },
        )
        return response["message"]["content"]

    def generate_json(self, prompt: str, schema: dict) -> str:
        """Generate a response constrained by a JSON schema."""
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=schema,
            options={
                "temperature": 0,
                "num_predict": 128,
            },
        )
        return response["message"]["content"]

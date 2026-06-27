from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


class SBERTEmbedder:
    """
    Generate embeddings using a pretrained SBERT model.
    """

    def __init__(
        self,
        input_dir: str,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.input_dir = Path(input_dir)

        print("Loading SBERT Model...")

        self.model = SentenceTransformer(model_name)

        print("Model Loaded Successfully.\n")

    def generate_embedding(self, text: str):
        """
        Generate embedding for one chunk.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()

    def process_file(self, json_file: Path):
        """
        Read one chunk file and generate embeddings.
        """

        with open(json_file, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        print(f"Generating embeddings for {json_file.name}")

        for chunk in chunks:

            chunk["embedding"] = self.generate_embedding(
                chunk["text"]
            )

        return chunks

    def process_all_files(self):
        """
        Generate embeddings for every chunk file.
        """

        all_chunks = []

        json_files = list(self.input_dir.glob("*.json"))

        for file in json_files:

            chunk_data = self.process_file(file)

            all_chunks.extend(chunk_data)

        print("\nAll embeddings generated successfully.\n")

        return all_chunks
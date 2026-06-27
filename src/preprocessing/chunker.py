from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Splits extracted text files into smaller chunks
    for semantic search.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk_text(self, text: str):
        """
        Split a single text into chunks.
        """

        return self.text_splitter.split_text(text)

    def process_file(self, txt_file: Path):
        """
        Read one txt file, split it into chunks,
        and save as JSON.
        """

        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = self.chunk_text(text)

        chunk_data = []

        for index, chunk in enumerate(chunks):

            chunk_data.append({
                "chunk_id": index + 1,
                "source": txt_file.stem,
                "text": chunk
            })

        output_file = self.output_dir / f"{txt_file.stem}.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(chunk_data, file, indent=4, ensure_ascii=False)

        print(f"✓ Chunked: {txt_file.name}")

    def process_all_files(self):
        """
        Process every extracted text file.
        """

        txt_files = list(self.input_dir.glob("*.txt"))

        if not txt_files:
            print("No text files found.")
            return

        print(f"\nFound {len(txt_files)} text files.\n")

        for txt_file in txt_files:
            self.process_file(txt_file)

        print("\n✅ Chunking Completed.")
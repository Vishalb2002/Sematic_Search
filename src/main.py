from loaders.pdf_loader import PDFLoader
from preprocessing.chunker import TextChunker


def main():

    loader = PDFLoader(
        input_dir="data/raw/hand_gestures",
        output_dir="data/processed/hand_gestures"
    )

    loader.process_all_pdfs()

      # -----------------------------
    # Step 2 : Chunk Text
    # -----------------------------
    chunker = TextChunker(
        input_dir="data/processed",
        output_dir="data/chunks"
    )

    chunker.process_all_files()


if __name__ == "__main__":
    main()
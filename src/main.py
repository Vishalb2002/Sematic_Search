from loaders.pdf_loader import PDFLoader
from preprocessing.chunker import TextChunker
from embeddings.embedder import SBERTEmbedder
from database.vector_store import VectorStore  
from retrieval.retriever import Retriever 

def main():

    # loader = PDFLoader(
    #     input_dir="data/raw/hand_gestures",
    #     output_dir="data/processed/hand_gestures"
    # )

    # loader.process_all_pdfs()

      # -----------------------------
    # Step 2 : Chunk Text
    # -----------------------------
    # chunker = TextChunker(
    #     input_dir="data/processed/hand_gestures",
    #     output_dir="data/chunks/hand_gestures"
    # )

    # chunker.process_all_files()

     # STEP 3
    # embedder = SBERTEmbedder(
    #     input_dir="data/chunks/hand_gestures"
    # )

    # embedded_chunks = embedder.process_all_files()

    # print(embedded_chunks[5])

    #  # --------------------------
    # # STEP 4
    # # --------------------------

    # vector_store = VectorStore()

    # vector_store.add_embeddings(
    #     embedded_chunks
    # )

    # print(
    #     f"\nTotal Chunks in DB : {vector_store.total_chunks()}"
    # )


    retriever = Retriever()

    results = retriever.search(
    "Which papers use Vision Transformer?"
    )

    print(results)

    context = retriever.build_context(results)

    print(context)


if __name__ == "__main__":
    main()
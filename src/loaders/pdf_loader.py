import fitz  # PyMuPDF
from pathlib import Path


class PDFLoader:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initializes the PDF Loader.

        Args:
            input_dir (str): Folder containing PDF files.
            output_dir (str): Folder where extracted text files will be saved.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            str: Extracted text.
        """

        document = fitz.open(pdf_path)

        text = ""

        for page in document:
            text += page.get_text("text") + "\n"

        document.close()

        return text

    def process_all_pdfs(self):
        """
        Process every PDF in the input directory and save as .txt files.
        """

        pdf_files = list(self.input_dir.glob("*.pdf"))

        if not pdf_files:
            print("❌ No PDF files found.")
            return

        print(f"\nFound {len(pdf_files)} PDF files.\n")

        for pdf_file in pdf_files:

            print(f"Processing: {pdf_file.name}")

            extracted_text = self.extract_text_from_pdf(pdf_file)

            output_file = self.output_dir / f"{pdf_file.stem}.txt"

            with open(output_file, "w", encoding="utf-8") as file:
                file.write(extracted_text)

            print(f"Saved: {output_file.name}\n")

        print("✅ All PDFs processed successfully.")
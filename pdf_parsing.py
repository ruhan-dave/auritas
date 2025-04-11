import os
import fitz  # PyMuPDF

class PDFTextExtractor:
    def __init__(self, input_pdf_path: str, output_dir: str):
        self.input_pdf_path = input_pdf_path
        self.output_dir = output_dir
        self.output_txt_path = self._generate_output_path()

    def _generate_output_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.input_pdf_path))[0]
        return os.path.join(self.output_dir, f"{base_name}.txt")

    def extract_text(self) -> str:
        if not os.path.isfile(self.input_pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.input_pdf_path}")

        doc = fitz.open(self.input_pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        doc.close()
        return full_text

    def save_text(self) -> str:
        text = self.extract_text()
        with open(self.output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return self.output_txt_path

# Example Usage:
# extractor = PDFTextExtractor("./sample.pdf", "./parsed_texts")
# output_file = extractor.save_text()
# print(f"✅ Text saved to: {output_file}")

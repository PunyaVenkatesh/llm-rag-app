from PyPDF2 import PdfReader

def load_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                print(f"⚠️ Warning: Page {i+1} seems empty or has no extractable text.")
            text += page_text + "\n"
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("No extractable text found in the PDF.")
        return cleaned_text
    except Exception as e:
        return f"Error reading PDF: {e}"

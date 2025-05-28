import os
from docx import Document
import PyPDF2

def load_document(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if ext == ".pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == ".docx":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        return uploaded_file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type")

def chunk_text(text, max_tokens=500):
    """
    Splits text into chunks of approximately `max_tokens` words each.
    """
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

"""
Parsing utilities: extract raw text from resume PDFs and JD text blocks.
The actual structuring into skills/experience/etc. is done by the
Parser Agent (LLM-based) in agents.py — this module only handles
raw text extraction so the agent has clean input to work with.
"""
import io
import pdfplumber


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file's bytes."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    """Dispatch based on file extension. Supports .pdf and plain text."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    # Fallback: treat as plain text (txt, md, etc.)
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")

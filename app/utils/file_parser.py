import io
import logging
import os
from pypdf import PdfReader
import docx

logger = logging.getLogger("recruitai-backend.file_parser")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text content from PDF file bytes.
    """
    try:    
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Could not parse PDF file: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extracts text content from DOCX file bytes.
    """
    try:
        docx_file = io.BytesIO(file_bytes)
        doc = docx.Document(docx_file)
        paragraphs = [p.text for p in doc.paragraphs]
        # Include text inside tables if any
        table_text = []
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    table_text.append(cell.text)
        
        full_text = "\n".join(paragraphs + table_text)
        return full_text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise ValueError(f"Could not parse DOCX file: {e}")

def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Extracts text from PDF resume file.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF is supported.")

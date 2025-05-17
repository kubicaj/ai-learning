from pypdf import PdfReader

def get_pdf_content(pdf_path: str) -> str:
    """
    Read PDF

    Args:
        - pdf_path: Path to PDF

    Returns:
        - text content of PDF
    """
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    return pdf_text
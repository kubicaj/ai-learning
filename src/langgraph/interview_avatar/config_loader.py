"""
Module which will load all config (not only regular config but also some prompts and CV etc
"""
from pypdf import PdfReader


def _load_md_resource_file(file_name: str) -> str:
    """
    Load position description

    Return:
        str representation of position desc
    """
    with open(f'resources/{file_name}', 'r') as file:
        return file.read()

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

CANDIDATE_CV = get_pdf_content("resources/cv_candidate.pdf")
POSITION_DESCRIPTION = _load_md_resource_file("open_positions/position_description.md")
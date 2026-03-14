import pdfplumber
import re
from pathlib import Path

def clean_text(text: str) -> str:
    # Remove citation brackets like [1], [a], [23]
    text = re.sub(r"\[[^\]]*\]", "", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def ingest_pdf(path):
    file_path = Path(path)
    BASE_DIR = Path(__file__).resolve().parent.parent
    pdf_folder = BASE_DIR / file_path
    all_paragraphs = []
    for pdf_file in pdf_folder.glob("*.pdf"):
        team_name = pdf_file.stem  # e.g., devops, docker
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_paragraphs = re.split(r'\n\s*\n|[-]{5,}', text)
                    for p in raw_paragraphs:
                        clean_p = clean_text(p.strip())
                        if clean_p and len(clean_p) > 50:  # ignore tiny fragments
                            all_paragraphs.append({
                                "team": team_name,
                                "text": clean_p
                        })
                    
    return all_paragraphs
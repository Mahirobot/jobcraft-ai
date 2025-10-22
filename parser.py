# resume_parser.py

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
import PyPDF2
from docx import Document

import summarizer
from config import CONFIG
from database import embed_resume_text, query_jobs_by_embedding

logger = logging.getLogger(__name__)


class ResumeParsingError(Exception):
    """Raised when resume parsing fails."""

    pass


def _parse_pdf(file_path: str) -> str:
    """Parse PDF using pdfplumber; fall back to PyPDF2 on failure."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page is not None:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    except Exception as e:
        logger.warning(
            f"pdfplumber failed for {file_path}: {e}. Falling back to PyPDF2."
        )
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e2:
            raise ResumeParsingError(f"Both pdfplumber and PyPDF2 failed: {e2}") from e2

    return text


def _parse_docx(file_path: str) -> str:
    """Parse DOCX using python-docx."""
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        raise ResumeParsingError(f"DOCX parsing failed for {file_path}: {e}")


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize."""
    return " ".join(text.split())


def _is_likely_resume_or_cover_letter(text: str, threshold: int = 4) -> bool:
    """
    Returns True if the text is likely a resume or cover letter.
    Uses a refined set of structural and contextual keywords.
    """
    resume_keywords = {
        # Core resume sections
        "work experience",
        "professional experience",
        "employment history",
        "education",
        "academic background",
        "qualifications",
        "skills",
        "technical skills",
        "core competencies",
        "certifications",
        "licenses",
        "awards",
        "honors",
        "projects",
        "professional projects",
        "key achievements",
        "summary",
        "professional summary",
        "career objective",
        "references available upon request",
        "contact",
        "email",
        "experience",
        "professional experience"
        # Cover letter indicators
        "dear hiring manager",
        "dear recruiter",
        "to the hiring team",
        "i am writing to express interest",
        "i am excited to apply",
        "enclosed is my resume",
        "my resume is attached",
        "thank you for considering my application",
        "sincerely,",
        "yours faithfully",
        "best regards,",
        # Common resume/contact formatting (used cautiously)
        "linkedin.com/in/",
        "github.com/",  # more specific URLs
    }

    text_lower = text.lower()
    matches = sum(1 for keyword in resume_keywords if keyword in text_lower)
    return matches >= threshold


def _passes_basic_checks(text: str) -> bool:
    """
    Basic sanity checks: length and printability.
    Resumes and cover letters are typically 100–5000 chars.
    """
    stripped = text.strip()
    if len(stripped) < 100 or len(stripped) > 5000:
        return False
    # Ensure mostly printable content
    if not stripped:
        return False
    printable_ratio = sum(c.isprintable() or c.isspace() for c in stripped) / len(
        stripped
    )
    return printable_ratio >= 0.95


def parse_resume_file(file_path: str) -> Dict[str, str]:
    """
    Parse resume from PDF or DOCX and return extracted plain text.

    Returns:
        {"file_path": str, "extracted_text": str}
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise ResumeParsingError(f"File does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    raw_text = ""

    if suffix == ".pdf":
        raw_text = _parse_pdf(str(file_path))
    elif suffix == ".docx":
        raw_text = _parse_docx(str(file_path))
    else:
        raise ResumeParsingError(
            f"Unsupported file type: {suffix}. Only .pdf and .docx are supported."
        )

    cleaned_text = _clean_text(raw_text)
    if not _is_likely_resume_or_cover_letter(cleaned_text) or not _passes_basic_checks(
        cleaned_text
    ):
        return {"file_path": str(file_path), "extracted_text": "Not a Resume."}
    if not cleaned_text.strip():
        raise ResumeParsingError(f"No text extracted from {file_path}")

    logger.info(f"Successfully parsed resume: {file_path}")
    return {"file_path": str(file_path), "extracted_text": cleaned_text}


def process_resume(file_path: str) -> Dict[str, object]:
    """
    Parse resume and generate embedding for querying.

    Returns:
        {"text": str, "embedding": List[float]}
    """
    try:
        parsed = parse_resume_file(file_path)
        embedding = embed_resume_text(parsed["extracted_text"])
        return {"text": parsed["extracted_text"], "embedding": embedding}
    except Exception as e:
        logger.error(f"Failed to process resume {file_path}: {e}")
        raise ResumeParsingError(f"Resume processing failed: {e}") from e


def match_jobs_for_resume(resume_file_path: str, n_results: int = 5):
    # Step 1: Parse and embed resume
    resume_data = process_resume(resume_file_path)

    # Step 2: Query ChromaDB using the embedding
    results = query_jobs_by_embedding(
        embedding=resume_data["embedding"], n_results=n_results
    )

    return {
        "resume_text": resume_data["text"],
        "matches": results,  # assumed to be list of JobEntry or similar
    }


# to test out parser
def format_job_matches(data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw ChromaDB query results into a clean, readable list of job dictionaries.

    Args:
        results: Dict with keys: 'metadatas', 'distances', 'ids', 'collection_names'

    Returns:
        List of dicts, each containing:
            - rank (int)
            - title (str)
            - company (str)
            - location (str)
            - tags (str)
            - distance (float)
            - job_id (str)
            - source_url (str)
    """
    print(f"Resume: \n{data['resume_text']}\n")
    results = data["matches"]
    if not results.get("ids"):
        return []

    formatted = []
    for i, (metadata, distance, job_id, collection) in enumerate(
        zip(
            results["metadatas"],
            results["distances"],
            results["ids"],
            results.get("collection_names", [None] * len(results["ids"])),
        )
    ):
        formatted.append(
            {
                "rank": i + 1,
                "title": metadata.get("title", "").strip(),
                "company": metadata.get("company", "").strip() or "N/A",
                "location": metadata.get("location", "").strip() or "N/A",
                "tags": metadata.get("tags", "").strip(),
                "distance": round(float(distance), 4),
                "job_id": job_id,
                "source_url": metadata.get("source_url", "").strip(),
                "collection": collection or "N/A",
                "description": metadata.get("description"),
                "summary": summarizer.summarize_job_description(
                    metadata["description"], max_sentences=3
                ),
            }
        )

        print(
            f"{formatted[-1]['rank']}. {formatted[-1]['title']} @ {formatted[-1]['company']} ({formatted[-1]['location']})"
        )
        print(f"   Tags: {formatted[-1]['tags']}")
        print(
            f"   Relevance: {formatted[-1]['distance']:.4f} | Apply: {formatted[-1]['source_url']}"
        )
        # print(f"   Description: \n{formatted[-1]['description']}\n")
        print(f"   Summary: \n{formatted[-1]['summary']}\n")
    return formatted


# results = parser.match_jobs_for_resume("Resume_Mahira.pdf", 10)

# format_job_matches(results)

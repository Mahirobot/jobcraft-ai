import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
# Match logging style from matcher.py
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = Groq is not None and GROQ_API_KEY is not None


def _call_groq(prompt: str) -> Optional[str]:
    if not USE_GROQ:
        logger.warning("Groq not configured; skipping LLM call.")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq API error during cover letter generation: {e}")
        return None


def _fallback_cover_letter(resume_text: str, job_metadata: Dict[str, Any]) -> str:
    company = job_metadata.get("company", "the company")
    title = job_metadata.get("title", "the role")
    return (
        f"Dear Hiring Team at {company},\n\n"
        f"I am writing to express my interest in the {title} position. "
        "My resume, attached for your review, details my relevant experience and skills. "
        "I am confident that my background aligns well with your requirements and I would welcome the opportunity to contribute to your team.\n\n"
        "Thank you for your consideration."
    )


def generate_personalized_cover_letter(
    resume_text: str,
    job_metadata: Dict[str, Any],
    example_cover_letters: Optional[List[str]] = None,
) -> str:
    title = job_metadata.get("title", "")
    company = job_metadata.get("company", "")
    description = job_metadata.get("description", "")

    # Build style context if examples provided
    style_block = ""
    if example_cover_letters:
        examples = "\n---\n".join(example_cover_letters[:2])  # Use up to 2
        style_block = (
            f"USER'S WRITING STYLE (from past cover letters):\n{examples}\n---\n"
        )

    prompt = f"""You are an expert career writer. Write a cover letter for the following job using the candidate's resume.

JOB:
Title: {title}
Company: {company}
Description: {description}

RESUME:
{resume_text}

{style_block}Instructions:
- Write 180–250 words.
- Sound like the user (match tone, style, and voice from examples if given).
- Highlight 2–3 specific experiences from the resume that match the job.
- Do NOT invent skills or roles not in the resume.
- Address the letter to "Hiring Team at {company}".
- Output ONLY the letter. No subject line, no signature."""

    # Try Groq
    response = _call_groq(prompt)
    if response:
        return response

    # Fallback if LLM unavailable
    logger.warning("Falling back to generic cover letter.")
    return _fallback_cover_letter(resume_text, job_metadata)

from dotenv import load_dotenv

load_dotenv()
import json
import logging
import os
from parser import match_jobs_for_resume
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq
import requests
from config import CONFIG

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

USE_GROQ = Groq is not None and GROQ_API_KEY is not None
import tiktoken


def truncate_text_to_tokens(text: str, max_tokens: int = 2000) -> str:
    """Truncate text to a max number of tokens using Llama-compatible tokenizer."""
    if not text:
        return ""
    encoding = tiktoken.get_encoding("cl100k_base")  # good proxy for Llama 3
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens]) + " [...] (truncated)"


def _build_prompt(resume_text: str, jobs: list) -> str:
    # ✅ Truncate resume to ~1500 tokens
    safe_resume = truncate_text_to_tokens(resume_text, max_tokens=1500)

    job_blocks = []
    for job in jobs:
        title = job.get("title", "N/A")
        company = job.get("company", "N/A")
        desc = job.get("description", "No description")
        location = job.get("location", "N/A")

        # ✅ Truncate each job description to ~400 tokens (5 jobs × 400 = 2000)
        safe_desc = truncate_text_to_tokens(desc, max_tokens=400)

        block = f"""Job: {title}
Company: {company}
Location: {location}
Description: {safe_desc}"""
        job_blocks.append(block)

    jobs_section = "\n\n---\n\n".join(job_blocks)

    prompt = f"""You are an expert recruiter. Analyze the candidate's resume and match it to the following job postings.

Resume:
\"\"\"
{safe_resume}
\"\"\"

Jobs:
\"\"\"
{jobs_section}
\"\"\"

For each job, provide:
- job_title (copy exactly from input)
- match_score (0–10, be strict)
- match_reason (1–2 sentences)
- skill_gaps (list of missing skills, or empty list)
- apply_link (if available in job data) [YOU MUST PROVIDE THIS]

Output ONLY a JSON array of objects with those keys. No other text."""

    return prompt


def _call_llm(prompt: str, st) -> str:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return "[]"
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": CONFIG['llm_model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 6000,
                "response_format": {"type": "json_object"}
            },
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        # Log to Streamlit UI for debugging
        # st.error(f"⚠️ Groq API failed (via requests): {e}")
        return "[]"


def generate_job_match_report(resume_file_path: str, n_results: int = 10) -> Dict:
    """
    Match jobs in batches of 5 to avoid LLM token limits.
    Returns sorted matches by match_score (descending).
    """
    try:
        result = match_jobs_for_resume(
            resume_file_path, n_results * 2
        )  # Fetch more to allow top-N after scoring
        resume_text = result["resume_text"]
        raw_jobs = result["matches"]["metadatas"]
        if not raw_jobs:
            return {"error": "No job matches found."}

        all_matches = []
        batch_size = 5

        # Process in batches
        for i in range(0, len(raw_jobs), batch_size):
            batch = raw_jobs[i : i + batch_size]
            prompt = _build_prompt(resume_text, batch)
            raw_response = _call_llm(prompt)
            try:
                parsed = json.loads(raw_response)
                if isinstance(parsed, list):
                    # Merge LLM output with full job metadata
                    for j, llm_job in enumerate(parsed):
                        if j >= len(batch):
                            break
                        full_job = batch[j].copy()
                        full_job.update(
                            {
                                "match_score": float(llm_job.get("match_score", 0)),
                                "match_reason": llm_job.get("match_reason", ""),
                                "skill_gaps": llm_job.get("skill_gaps", []),
                                "apply_link": llm_job.get(
                                    "apply_link", full_job.get("source_url", "")
                                ),
                            }
                        )
                        all_matches.append(full_job)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Batch {i//batch_size + 1} JSON parse failed: {e}")
                continue

        if not all_matches:
            return {"error": "No valid matches returned from LLM."}

        # Sort by match_score descending
        all_matches.sort(key=lambda x: x.get("match_score", 0), reverse=True)

        # Return top n_results
        return {"matches": all_matches[:n_results]}

    except Exception as e:
        logger.exception("Unexpected error in generate_job_match_report")
        return {"error": f"Unexpected error: {str(e)}"}


# formatter.py


def format_job_match_report(report: dict) -> str:
    if "error" in report:
        return f"Error: {report['error']}\n"

    if not report.get("matches"):
        return "No job matches found.\n"

    lines = ["### Top Job Matches\n"]
    for i, job in enumerate(report["matches"], 1):
        title = job.get("job_title", "N/A")
        company = job.get("company", "N/A")
        score = job.get("match_score", 0)
        reason = job.get("match_reason", "").strip()
        gaps = job.get("skill_gaps", [])
        link = job.get("apply_link", "").strip()

        lines.append(f"\n#### {i}. **{title}**")
        lines.append(f"\n**Company**: {company}")
        lines.append(f"\n**Match Score**: {score} / 10")
        lines.append(f"\n**Why it matches**: {reason}")

        if gaps:
            gap_list = "\n  - ".join(gaps)
            lines.append(f"\n**Skill Gaps**:\n  - {gap_list}")
        else:
            lines.append("\n**Skill Gaps**: None")

        if link:
            # Clean extra whitespace in URL
            clean_link = link.split()[0] if link else ""
            lines.append(f"\n**Apply**: [View Job]({clean_link})")

        lines.append("---")

    return "\n".join(lines).strip()

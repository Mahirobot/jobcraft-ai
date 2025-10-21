from dotenv import load_dotenv

load_dotenv()
import json
import logging
import os
from parser import match_jobs_for_resume
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq

from config import CONFIG

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

USE_GROQ = Groq is not None and GROQ_API_KEY is not None


def _build_prompt(resume_text: str, jobs: List[Dict[str, Any]]) -> str:
    job_blocks = []
    for i, job in enumerate(jobs):
        block = (
            f"{i+1}. TITLE: {job['title']}\n"
            f"    COMPANY: {job['company']}\n"
            f"    LOCATION: {job['location']}\n"
            f"    POSTED: {job['posted_date']}\n"
            f"    DESCRIPTION:\n"
            f"    {job['description']}\n"
            f"    APPLY: {job['source_url']}"
        )
        job_blocks.append(block)

    jobs_section = "\n\n".join(job_blocks)

    return (
        "You are an expert career advisor. Output your response STRICTLY as a JSON object with a key 'jobs' containing an array of job matches. Do NOT include any other text, markdown, or explanation.. Analyze the following resume and job listings. For each job, provide:\n"
        "- A match score from 0 to 10\n"
        "- A 2–3 sentence explanation of why it matches (or doesn’t), referencing specific skills or experiences\n"
        "- A list of missing skills or gaps (if any)\n\n"
        f'RESUME:\n"""\n{resume_text}\n"""\n\n'
        f"JOBS:\n{jobs_section}\n\n"
        "Output your response strictly as a JSON array. Do not include any other text.\n"
        "Format:\n"
        "[\n"
        "  {\n"
        '    "job_title": "...",\n'
        '    "company": "...",\n'
        '    "match_score": 8.5,\n'
        '    "match_reason": "...",\n'
        '    "skill_gaps": ["...", "..."],\n'
        '    "apply_link": "..."\n'
        "  },\n"
        "  ...\n"
        "]"
    )


def _call_llm(prompt: str) -> str:
    if USE_GROQ:
        client = Groq(api_key=GROQ_API_KEY)
        try:
            response = client.chat.completions.create(
                model=CONFIG["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=6000,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq API error: {e}")
            logger.debug(f"Raw LLM response: {response}")  # add this

    # Fallback: return empty JSON array if no LLM available
    logger.error("No LLM backend available")
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

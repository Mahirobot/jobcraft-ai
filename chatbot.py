# chatbot.py
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = GROQ_API_KEY is not None


def _call_groq(messages: List[Dict[str, str]]) -> str:
    if not USE_GROQ:
        return "I'm sorry, but the AI assistant is currently unavailable. Please try again later."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Apologies, I encountered an error: {str(e)}"


def get_chatbot_response(
    resume_text: str,
    job_metadata: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    cover_letter: str = "",
) -> str:
    title = job_metadata.get("job_title", "this role")
    company = job_metadata.get("company", "the company")
    description = job_metadata.get("description", "No description available.")
    reason = job_metadata.get("match_reason", "")
    gaps = job_metadata.get("skill_gaps", [])

    system_prompt = f"""You are an expert career advisor specializing in job applications. 
You are helping a candidate apply for:
- Role: {title}
- Company: {company}
- Why it matches: {reason}
- Skill gaps: {', '.join(gaps) if gaps else 'None'}

The candidate's resume:
\"\"\"
{resume_text}
\"\"\"

Job description:
\"\"\"
{description}
\"\"\"

Guidelines:
- Only discuss resume improvement, cover letters, interview prep, or application strategy.
- NEVER invent facts not in the resume or job description.
- Be encouraging, specific, and actionable.
- If asked to write a cover letter, draft a concise 150–200 word version.
- Keep responses under 3 sentences unless writing a cover letter.
"""
    if cover_letter:
        system_prompt += f'\n\nUSER\'S COVER LETTER STYLE:\n"""\n{cover_letter}\n"""\n'

    # Build message history (limit to last 5 exchanges to avoid token limits)
    messages = [{"role": "system", "content": system_prompt}]
    recent_history = chat_history[-10:]  # Last 5 user+assistant pairs
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    return _call_groq(messages)

# chatbot.py — updated version

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq
import requests
# ✅ Import your cover letter generator
from cover_letter import generate_personalized_cover_letter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = GROQ_API_KEY is not None


def _call_groq(messages: List[Dict[str, str]]) -> str:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return "I'm sorry, but the AI assistant is currently unavailable."
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Apologies, I encountered an error: {str(e)}"


def get_chatbot_response(
    resume_text: str,
    job_metadata: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    cover_letter: str = "",
) -> str:
    # ✅ Check if user is asking for cover letter generation
    if chat_history and chat_history[-1]["role"] == "user":
        user_query = chat_history[-1]["content"].lower()
        keywords = ["cover letter", "coverletter", "application letter"]
        actions = ["write", "generate", "create", "draft", "personalize", "make"]

        if any(kw in user_query for kw in keywords) and any(
            act in user_query for act in actions
        ):
            # ✅ Use your dedicated cover letter generator
            try:
                examples = [cover_letter] if cover_letter else []
                personalized = generate_personalized_cover_letter(
                    resume_text=resume_text,
                    job_metadata=job_metadata or {},
                    example_cover_letters=examples,
                )
                return f"Here’s your tailored cover letter:\n\n{personalized}"
            except Exception as e:
                return f"I tried to generate a cover letter, but ran into an issue: {str(e)}. Would you like me to try a simpler version?"

    # --- Fallback to normal chat ---
    if not job_metadata:
        job_metadata = {}
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
- If asked to write a cover letter, you may do so — but prefer using the candidate's style if provided.
- Keep responses under 3 sentences unless writing a cover letter.
"""
    if cover_letter:
        system_prompt += f'\n\nUSER\'S COVER LETTER STYLE:\n"""\n{cover_letter}\n"""\n'

    messages = [{"role": "system", "content": system_prompt}]
    recent_history = chat_history[-10:]
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    return _call_groq(messages)

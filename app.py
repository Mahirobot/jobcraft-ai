import json
import os
import tempfile
import time
from parser import embed_resume_text, parse_resume_file, process_resume
from pathlib import Path

import streamlit as st
from json_repair import repair_json

from chatbot import get_chatbot_response
from config import CONFIG
from cover_letter import generate_personalized_cover_letter
from database import query_jobs_by_embedding
from matcher import _build_prompt, _call_llm, format_job_match_report

import chromadb
debug_client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
debug_collections = debug_client.list_collections()

def stream_text(text: str, delay: float = 0.005):
    """
    Generator that yields characters one by one to simulate typing.
    Usage: st.write_stream(stream_text("Hello!"))
    """
    for char in text:
        yield char
        time.sleep(delay)

# set_page_config MUST be the FIRST Streamlit command
st.set_page_config(page_title="AI Resume-Job Matcher", layout="wide")
os.environ["CHROMA_TELEMETRY"] = "false"
st.write("🔍 DEBUG: Available collections:", [c.name for c in debug_collections])

# ---------------------------
# Custom CSS for dark theme + job cards + minimizable chat
# ---------------------------
st.markdown(
    """
<style>
    /* Dark theme */
    body {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    .job-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 320px;
        max-height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }
    .match-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
        color: white;
        background-color: #4CAF50;
        margin-top: 8px;
        align-self: flex-start;
    }
    .chat-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #121212;
        border-top: 1px solid #333;
        padding: 16px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.5);
        z-index: 100;
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        color: #ffffff;
    }
    .chat-messages {
        max-height: 200px;
        overflow-y: auto;
        padding: 8px;
        background: #1a1a1a;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.95em;
        color: #ffffff;
    }
    .user-msg {
        text-align: right;
        color: #4da6ff;
        margin: 4px 0;
    }
    .bot-msg {
        text-align: left;
        color: #ffffff;
        margin: 4px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        padding: 8px 16px;
        margin-top: 8px;
        background: #2a2a2a;
        color: white;
        border: 1px solid #444;
    }
    .stButton>button:hover {
        background: #333;
        border-color: #555;
    }
    .apply-button {
        background-color: #007BFF !important;
        color: white !important;
    }
    .divider {
        margin: 20px 0;
        height: 1px;
        background: #333;
    }
    .job-card h4 {
        margin: 0 0 8px 0;
    }
    .job-card p {
        margin: 4px 0;
        font-size: 0.95em;
        color: #ccc;
    }

    /* Minimal file uploader — hide drag/drop, show only icon */
    .stFileUploader {
        display: inline-block;
        padding: 0;
        margin: 0;
    }
    .stFileUploader label {
        background: transparent !important;
        border: none !important;
        padding: 6px 12px !important;
        font-size: 1.2em !important;
        cursor: pointer !important;
        color: #4da6ff !important;
        transition: color 0.2s ease, transform 0.2s ease;
        border-radius: 4px;
    }
    .stFileUploader label:hover {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
        transform: scale(1.05);
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"],
    .stFileUploader div[data-testid="stFileUploaderInner"] {
        display: none !important;
    }

    /* Floating chat toggle button */
    .chat-toggle {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 200;
    }
    .chat-toggle button {
        background: #1e1e1e;
        border: 1px solid #444;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        font-size: 1.4em;
        color: #4da6ff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .chat-toggle button:hover {
        background: #2a2a2a;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helper Functions
# ---------------------------


def extract_json(text: str) -> str:
    """Extract JSON array from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1].strip(" `\n")
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _call_llm_with_retry(prompt: str, max_retries=1) -> str:
    """Call LLM with JSON validation retry."""
    for attempt in range(max_retries + 1):
        raw = _call_llm(prompt)
        try:
            parsed = json.loads(raw)
            if (
                isinstance(parsed, dict)
                and "jobs" in parsed
                and isinstance(parsed["jobs"], list)
            ):
                return raw
        except:
            pass
        if attempt < max_retries:
            prompt += "\nIMPORTANT: Your output MUST be valid JSON. Do not add any extra text."
    return raw


# ---------------------------
# Session State Initialization
# ---------------------------
if "matches" not in st.session_state:
    st.session_state.matches = []
if "raw_matches" not in st.session_state:
    st.session_state.raw_matches = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = ""
if "max_matches" not in st.session_state:
    st.session_state.max_matches = CONFIG.get("max_result_to_display", 5)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_job" not in st.session_state:
    st.session_state.active_job = None
if "chat_minimized" not in st.session_state:
    st.session_state.chat_minimized = True
if "cover_letter_processed" not in st.session_state:
    st.session_state.cover_letter_processed = False
if "trigger_matching" not in st.session_state:
    st.session_state.trigger_matching = False
# ✅ NEW: Store uploaded cover letter text without auto-processing
if "uploaded_cover_letter_text" not in st.session_state:
    st.session_state.uploaded_cover_letter_text = None

# --- Sidebar with ALL filters ---
def extract_locations(jobs):
    return sorted(
        {job.get("location", "").strip() for job in jobs if job.get("location")}
    )


with st.sidebar:
    st.title("⚙️ Matching Settings")
    min_score = st.slider("Min Match Score", 0.0, 10.0, 5.0, 0.5)
    remote_only = False
    all_locations = extract_locations(st.session_state.raw_matches)
    selected_locations = st.multiselect("Location", options=all_locations)
    max_matches = st.slider("Max job matches", 1, 20, st.session_state.max_matches, 1)
    keywords = st.text_input(
        "Boost keywords", st.session_state.keywords, help="e.g., LLM, AWS, RAG"
    )
    if st.button("💾 Save"):
        st.session_state.max_matches = max_matches
        st.session_state.keywords = keywords
        st.success("Settings saved!")

# --- Header ---
st.title("🎯 JobCraft AI")

# --- Resume Uploader ONLY ---
st.markdown("<div style='margin-bottom: 20px;'>", unsafe_allow_html=True)
resume_file = st.file_uploader(
    "📄 Upload your resume (.pdf or .docx)",
    type=["pdf", "docx"],
    key="resume_uploader",
    help="Upload your resume to start matching jobs",
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Parse resume on upload (NO auto-matching) ---
if resume_file:
    if resume_file.type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        st.error("Invalid file type.")
    else:
        try:
            suffix = Path(resume_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resume_file.getvalue())
                resume_path = tmp.name

            resume_data = process_resume(resume_path)
            os.unlink(resume_path)

            if resume_data["text"] == "Not a Resume.":
                st.error(
                    "Please check your upload. Does not seem like a Resume/Cover letter."
                )
                st.session_state.resume_text = ""
            else:
                st.session_state.resume_text = resume_data["text"]
                st.success(
                    "✅ Resume uploaded! You can now ask questions or find matching jobs."
                )
        except Exception as e:
            st.error(f"Failed to parse resume: {e}")
            st.session_state.resume_text = ""

# --- Action buttons after resume upload ---
if st.session_state.resume_text:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True):
            st.session_state.trigger_matching = True
    with col2:
        if st.button("💬 Ask About My Resume", use_container_width=True):
            if not st.session_state.chat_history:
                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": "Hi! I can help you understand your resume, suggest improvements, or discuss job strategies. What would you like to know?",
                    }
                ]
            st.session_state.chat_minimized = False
            st.session_state.active_job = None  # General resume chat
            st.rerun()

# --- Manual Job Matching Logic ---
if st.session_state.trigger_matching and st.session_state.resume_text:

    def run_matching():
        try:
            base_text = st.session_state.resume_text
            injected = (
                base_text
                + "\n###Additional Keywords: "
                + st.session_state.keywords.strip()
                if st.session_state.keywords.strip()
                else base_text
            )

            embedding = embed_resume_text(injected)
            raw_jobs_result = query_jobs_by_embedding(
                embedding=embedding, n_results=st.session_state.max_matches * 2
            )
            st.write("🔍 DEBUG: Found", len(raw_jobs_result["metadatas"]), "jobs from ChromaDB")
            raw_jobs = raw_jobs_result["metadatas"]
            if not raw_jobs:
                st.session_state.matches = []
                st.session_state.raw_matches = []
                return

            all_matches = []
            batch_size = 5
            for i in range(0, len(raw_jobs), batch_size):
                batch = raw_jobs[i : i + batch_size]
                prompt = _build_prompt(base_text, batch)
                raw_resp = _call_llm_with_retry(prompt)
                clean_resp = extract_json(raw_resp)
                try:
                    parsed = json.loads(clean_resp)
                    if not isinstance(parsed, list):
                        repaired = repair_json(raw_resp, return_objects=True)
                        if isinstance(repaired, dict) and "jobs" in repaired:
                            parsed = repaired["jobs"]
                        else:
                            raise ValueError("Not a JSON array")
                    for j, llm_job in enumerate(parsed):
                        if j >= len(batch):
                            break
                        full = batch[j].copy()
                        full.update(
                            {
                                "job_title": llm_job.get("job_title")
                                or full.get("title", "Unknown"),
                                "match_score": float(llm_job.get("match_score", 0)),
                                "match_reason": llm_job.get("match_reason", ""),
                                "skill_gaps": llm_job.get("skill_gaps", []),
                                "apply_link": llm_job.get(
                                    "apply_link", full.get("source_url", "")
                                ),
                            }
                        )
                        all_matches.append(full)
                except Exception as e:
                    st.warning(f"Batch error: {e}")
                    continue

            all_matches.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            top = all_matches[: st.session_state.max_matches]
            st.session_state.matches = top
            st.session_state.raw_matches = top
            st.session_state.trigger_matching = False
        except Exception as e:
            st.error(
                f"Matching failed: Seems like we could not find a job match for you. :("
            )

    with st.spinner("Matching jobs..."):
        run_matching()

# --- Refresh button (only visible after matching) ---
if st.session_state.matches and st.session_state.resume_text:
    if st.button("🔄 Refresh Matches"):
        st.session_state.matches = []
        st.session_state.raw_matches = []
        st.session_state.chat_history = []
        st.session_state.active_job = None
        st.session_state.trigger_matching = True
        st.rerun()

# --- Apply Filters & Display Jobs ---
def filter_jobs(jobs, min_score, locations, remote_only):
    filtered = []
    for job in jobs:
        score = job.get("match_score", 0)
        loc = job.get("location", "")
        if score < min_score:
            continue
        if locations and loc not in locations:
            continue
        if remote_only and "remote" not in loc.lower():
            continue
        filtered.append(job)
    filtered.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return filtered


filtered_jobs = filter_jobs(
    st.session_state.raw_matches, min_score, selected_locations, remote_only
)

if filtered_jobs:
    st.subheader("🎯 Your Matches")
    cols = st.columns(3)
    for idx, job in enumerate(filtered_jobs):
        col = cols[idx % 3]
        with col:
            title = job.get("job_title", "N/A")
            company = job.get("company", "N/A")
            location = job.get("location", "N/A")
            score = job.get("match_score", 0)
            reason = job.get("match_reason", "")
            gaps = job.get("skill_gaps", [])
            link = job.get("apply_link", "#")

            badge_color = (
                "#4CAF50" if score >= 8 else "#FF9800" if score >= 6 else "#F44336"
            )
            badge_html = f'<span class="match-badge" style="background-color:{badge_color}">🟢 {score}/10</span>'

            with st.container():
                st.markdown(
                    f"""
                <div class="job-card">
                    {badge_html}
                    <h4>{title}</h4>
                    <p><strong>{company}</strong> • {location}</p>
                    <p><em>Why: {reason[:150]}{'...' if len(reason) > 150 else ''}</em></p>
                    <p><strong>Skill Gaps:</strong> {'None' if not gaps else ', '.join(gaps)}</p>
                """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "View Details",
                    key=f"view_{idx}",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_job_for_modal = job
                    st.session_state.show_modal = True

                if st.button(
                    "💬 Chat with Job Expert",
                    key=f"chat_modal_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.active_job = job
                    st.session_state.chat_history = [
                        {
                            "role": "assistant",
                            "content": f"Hi! I'm your job application advisor for {job['job_title']} at {job['company']}. How can I help you?",
                        }
                    ]
                    st.session_state.chat_minimized = False
                    st.session_state.cover_letter_processed = False
                    st.session_state.uploaded_cover_letter_text = None  # reset per job
                    st.rerun()

                if link and link != "#":
                    st.link_button("🚀 Apply", link, use_container_width=True)
                else:
                    st.button(
                        "🚀 Apply",
                        disabled=False,
                        use_container_width=True,
                        key=f"apply_disabled_{idx}",
                    )
                st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.matches is not None and not st.session_state.matches:
    st.warning("No jobs match your filters.")

# ---------------------------
# FLOATING CHAT TOGGLE BUTTON
# ---------------------------
col_toggle, _ = st.columns([1, 11])
with col_toggle:
    if st.session_state.chat_minimized:
        open_btn = st.button("💬", help="Open chat", key="open_chat")
    else:
        open_btn = st.button("🗨️", help="Minimize chat", key="close_chat")

if open_btn:
    st.session_state.chat_minimized = not st.session_state.chat_minimized
    st.rerun()

# ---------------------------
# MINIMIZABLE CHAT PANEL
# ---------------------------
if not st.session_state.chat_minimized:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-header"><strong>💬 Job Application Assistant</strong></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.active_job is None:
        if st.session_state.resume_text:
            st.info("You're chatting about your resume. Ask anything!")
        else:
            st.info("Upload a resume to start chatting.")
    else:
        job = st.session_state.active_job
        st.markdown(
            f"**Chatting about**: {job['job_title']} at {job['company']}",
            unsafe_allow_html=True,
        )

    # Display chat history
    with st.container():
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">You: {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bot-msg">Advisor: {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Input + Upload (only if resume is available)
    if st.session_state.resume_text:
        input_container = st.container()
        with input_container:
            st.markdown(
                """
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.4em; color: #4da6ff;">📄</span>
                <span style="font-size: 0.9em; color: #4da6ff; margin-left: 6px;">Upload your Cover Letter</span>
                <div style="flex-grow: 1;">
            """,
                unsafe_allow_html=True,
            )

            # ✅ Cover letter uploader: ONLY store text, no auto-generation
            cover_letter_file = st.file_uploader(
                "Upload cover letter",
                type=["pdf", "docx"],
                key="cover_letter_upload",
                label_visibility="collapsed",
                help="Upload your existing cover letter",
            )

            # Parse and store cover letter ONLY — no bot reply
            if cover_letter_file is not None:
                try:
                    suffix = Path(cover_letter_file.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(cover_letter_file.getvalue())
                        temp_path = tmp.name

                    cover_text = parse_resume_file(temp_path)["extracted_text"]
                    os.unlink(temp_path)
                    st.session_state.uploaded_cover_letter_text = cover_text
                    # ✅ Do NOT append to chat. Do NOT generate. Just store.
                except Exception as e:
                    st.session_state.uploaded_cover_letter_text = None
                    st.toast("❌ Failed to read cover letter.", icon="⚠️")

            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "", placeholder="Ask your job expert...", key="user_input"
                )
                submit = st.form_submit_button("Send")

                if submit and user_input:
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    bot_reply = get_chatbot_response(
                        resume_text=st.session_state.resume_text,
                        job_metadata=st.session_state.active_job,
                        chat_history=st.session_state.chat_history,
                        cover_letter=st.session_state.uploaded_cover_letter_text,  # pass to chatbot if needed
                    )
                    with st.chat_message("assistant"):
                        streamed_reply = st.write_stream(stream_text(bot_reply))
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": bot_reply}
                    )
                    st.rerun()

            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# JOB DETAIL MODAL
# ---------------------------
@st.dialog("Job Details")
def job_detail_modal(job):
    st.subheader(job.get("job_title", "N/A"))
    st.markdown(f"**Company**: {job.get('company', 'N/A')}")
    st.markdown(f"**Location**: {job.get('location', 'N/A')}")

    score = job.get("match_score", 0)
    badge_color = "#4CAF50" if score >= 8 else "#FF9800" if score >= 6 else "#F44336"
    st.markdown(
        f'<span class="match-badge" style="background-color:{badge_color}">🟢 {score}/10</span>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Why it matches**:\n\n{job.get('match_reason', '')}")

    gaps = job.get("skill_gaps", [])
    if gaps:
        st.markdown("**Skill Gaps**: " + ", ".join(gaps))
    else:
        st.markdown("**Skill Gaps**: None")

    apply_link = job.get("apply_link", "")
    if apply_link and apply_link != "#":
        st.link_button("🚀 Apply", apply_link)

    if st.button(
        "💬 Chat with Job Expert", key=f"chat_modal_in_detail", use_container_width=True
    ):
        st.session_state.active_job = job
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": f"Hi! I'm your job application advisor for {job['job_title']} at {job['company']}. How can I help you?",
            }
        ]
        st.session_state.chat_minimized = False
        st.session_state.cover_letter_processed = False
        st.session_state.uploaded_cover_letter_text = None
        st.rerun()


if "show_modal" in st.session_state and st.session_state.show_modal:
    job_detail_modal(st.session_state.selected_job_for_modal)
    st.session_state.show_modal = False

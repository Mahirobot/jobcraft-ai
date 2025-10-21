import re
from collections import Counter
from typing import Dict, List, Set

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import CONFIG

# Optional: preload model once if used frequently
_SUMMARIZER_MODEL = SentenceTransformer(CONFIG["summarizer_model"])


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract important keywords using simple heuristics:
    - Tech terms (capitalized words, acronyms like AWS, LLM, RAG)
    - Common job-related nouns (Python, Kubernetes, PostgreSQL, etc.)
    """
    # Clean and split
    words = re.findall(r"\b[A-Za-z0-9]{2,}\b", text)
    words = [w for w in words if not w.isdigit()]

    # Boost capitalized/acronym-like words (likely tech terms)
    scored = {}
    for w in words:
        base_score = 1
        if w[0].isupper() or w.isupper() or len(w) <= 4:
            base_score += 1
        if w.lower() in {
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "c++",
            "node",
            "react",
            "aws",
            "gcp",
            "kubernetes",
            "docker",
            "postgres",
            "mysql",
            "mongodb",
            "redis",
            "elasticsearch",
            "sagemaker",
            "langchain",
            "crewai",
            "llm",
            "rag",
            "ai",
            "ml",
            "mlops",
            "vector",
            "graphql",
            "rest",
            "api",
        }:
            base_score += 2
        scored[w.lower()] = scored.get(w.lower(), 0) + base_score

    # Return top-k unique keywords
    return [k for k, _ in Counter(scored).most_common(top_k)]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using basic punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def summarize_job_description(
    description: str, max_sentences: int = 3, include_keywords: bool = True
) -> Dict[str, any]:
    """
    Summarize a job description by selecting the most representative sentences
    using cosine similarity to the full text (centroid-based summarization).

    Returns:
        {
            "summary": str,
            "keywords": List[str] (if include_keywords=True)
        }
    """
    sentences = split_into_sentences(description)
    if len(sentences) <= max_sentences:
        summary = " ".join(sentences)
    else:
        model = _SUMMARIZER_MODEL
        # Encode all sentences and the full doc
        embeddings = model.encode(sentences + [description])
        doc_embedding = embeddings[-1]
        sent_embeddings = embeddings[:-1]

        # Compute similarity of each sentence to the full doc
        sims = cosine_similarity([doc_embedding], sent_embeddings)[0]

        # Pick top sentences by similarity
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[
            :max_sentences
        ]
        top_indices.sort()  # preserve original order
        summary = " ".join(sentences[i] for i in top_indices)

    result = {"summary": summary}
    if include_keywords:
        result["keywords"] = extract_keywords(description)

    return result

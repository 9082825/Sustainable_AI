# src/prompt_optimizer.py

import re
from typing import Dict


# Very small English stopword list (enough for ratios in our demo)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "of", "in", "on", "at", "for", "with", "to", "from", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its", "as", "about", "into", "over", "under",
    "you", "your", "i", "we", "they", "he", "she", "them",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str):
    # Very simple sentence splitter based on punctuation
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


def _tokenize_words(text: str):
    # Lowercase and keep only alphabetic-ish chunks
    text = text.lower()
    return re.findall(r"[a-z']+", text)


def _compute_complexity(prompt: str) -> Dict[str, float]:
    """
    Compute basic linguistic features for the given prompt.

    Returns a dict with keys:
      - tokens
      - avg_sentence_len
      - stopword_ratio
      - sections
    """
    text = _normalize_whitespace(prompt)
    sentences = _split_sentences(text)
    tokens = _tokenize_words(text)

    n_tokens = len(tokens)
    n_sentences = max(1, len(sentences))  # avoid division by zero
    n_stopwords = sum(1 for t in tokens if t in STOPWORDS)

    avg_sentence_len = n_tokens / n_sentences if n_sentences > 0 else 0.0
    stopword_ratio = n_stopwords / n_tokens if n_tokens > 0 else 0.0

    # For now, treat "sections" as number of sentences (you can refine later)
    sections = n_sentences

    return {
        "tokens": float(n_tokens),
        "avg_sentence_len": float(avg_sentence_len),
        "stopword_ratio": float(stopword_ratio),
        "sections": float(sections),
    }


def _semantic_similarity(original: str, simplified: str) -> float:
    """
    Very rough semantic similarity based on Jaccard overlap of content tokens.
    This is only for demonstration, not real semantics.
    """
    orig_tokens = set(_tokenize_words(original)) - STOPWORDS
    simp_tokens = set(_tokenize_words(simplified)) - STOPWORDS

    if not orig_tokens or not simp_tokens:
        return 0.0

    intersection = len(orig_tokens & simp_tokens)
    union = len(orig_tokens | simp_tokens)
    return intersection / union if union > 0 else 0.0


def optimize_prompt(prompt: str) -> dict:
    """
    Simple optimization:
    - If the prompt has more than 40 words, truncate and add "..."
    - Compute basic complexity metrics before/after
    - Estimate a crude semantic similarity score (0–1)

    Returns a dict that app.py expects:
    {
        "original_prompt": str,
        "simplified_prompt": str,
        "complexity_before": {...},
        "complexity_after": {...},
        "semantic_similarity": float,
    }
    """
    prompt = _normalize_whitespace(prompt)
    tokens = prompt.split()

    if len(tokens) > 40:
        simplified = " ".join(tokens[:40]) + " ..."
    else:
        simplified = prompt

    complexity_before = _compute_complexity(prompt)
    complexity_after = _compute_complexity(simplified)
    sim = _semantic_similarity(prompt, simplified)

    result = {
        "original_prompt": prompt,
        "simplified_prompt": simplified,
        "complexity_before": complexity_before,
        "complexity_after": complexity_after,
        "semantic_similarity": float(sim),
    }
    return result

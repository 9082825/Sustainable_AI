# src/prompt_optimizer.py

import re
from typing import Dict


# Small English stopword list – just for ratios
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
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


def _tokenize_words(text: str):
    text = text.lower()
    return re.findall(r"[a-z']+", text)


def _compute_complexity(prompt: str) -> Dict[str, float]:
    """
    Basic linguistic features:
      - tokens
      - avg_sentence_len
      - stopword_ratio
      - sections  (≈ #sentences)
    """
    text = _normalize_whitespace(prompt)
    sentences = _split_sentences(text)
    tokens = _tokenize_words(text)

    n_tokens = len(tokens)
    n_sentences = max(1, len(sentences))
    n_stopwords = sum(1 for t in tokens if t in STOPWORDS)

    avg_sentence_len = n_tokens / n_sentences if n_sentences > 0 else 0.0
    stopword_ratio = n_stopwords / n_tokens if n_tokens > 0 else 0.0
    sections = n_sentences

    return {
        "tokens": float(n_tokens),
        "avg_sentence_len": float(avg_sentence_len),
        "stopword_ratio": float(stopword_ratio),
        "sections": float(sections),
    }


def _semantic_similarity(original: str, simplified: str) -> float:
    """
    Very rough semantic similarity via Jaccard overlap of content tokens.
    Only for demo – not true semantics.
    """
    orig_tokens = set(_tokenize_words(original)) - STOPWORDS
    simp_tokens = set(_tokenize_words(simplified)) - STOPWORDS

    if not orig_tokens or not simp_tokens:
        return 0.0

    intersection = len(orig_tokens & simp_tokens)
    union = len(orig_tokens | simp_tokens)
    return intersection / union if union > 0 else 0.0


FILLER_PATTERNS = [
    r"\bplease\b",
    r"\bkindly\b",
    r"\bi would like you to\b",
    r"\bi want you to\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bjust\b",
]


def _simplify_text(prompt: str) -> str:
    """
    Light-weight prompt simplifier:
    - normalize whitespace
    - remove common filler phrases
    - keep sentences intact (NO hard truncation)
    """
    text = _normalize_whitespace(prompt)

    for pat in FILLER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # collapse multiple spaces created by removals
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def optimize_prompt(prompt: str) -> dict:
    """
    Main API called by app.py

    Returns:
      {
        "original_prompt": str,
        "simplified_prompt": str,
        "complexity_before": {...},
        "complexity_after": {...},
        "semantic_similarity": float in [0,1],
      }
    """
    prompt = _normalize_whitespace(prompt)

    simplified = _simplify_text(prompt)

    complexity_before = _compute_complexity(prompt)
    complexity_after = _compute_complexity(simplified)
    sim = _semantic_similarity(prompt, simplified)

    return {
        "original_prompt": prompt,
        "simplified_prompt": simplified,
        "complexity_before": complexity_before,
        "complexity_after": complexity_after,
        "semantic_similarity": float(sim),
    }

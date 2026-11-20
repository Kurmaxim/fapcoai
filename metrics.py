from collections import Counter
from itertools import combinations
import math
import os
from pathlib import Path
import re

from agents import SECTIONS

KEYWORDS = {
    "price": ("руб", "бюджет", "стоим", "цен"),
    "comfort": ("багаж", "мест", "расход", "вместим", "кресл"),
    "reliability": ("надежн", "обслуж", "запчаст", "ремонт"),
}
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ENCODERS = {}


def is_filled(section, text, min_chars=100):
    normalized = text.strip().lower().replace("ё", "е")
    return len(text.strip()) >= min_chars and any(word in normalized for word in KEYWORDS[section])


def coverage(sections, min_chars=100):
    filled = {section: any(is_filled(section, item["text"], min_chars)
                           for item in sections.get(section, [])) for section in SECTIONS}
    return {"coverage": sum(filled.values()), "coverage_ratio": sum(filled.values()) / 3,
            "filled_sections": filled}


def cosine(a, b):
    denominator = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    return max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b)) / denominator)) if denominator else 0.0


def load_encoder(cache_dir=None, revision=None):
    cache = cache_dir or os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("HF_HOME")
    cache = str(Path(cache).resolve())
    os.environ["HF_HOME"] = os.getenv("HF_HOME", cache)
    os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache
    key = (cache, revision)
    if key not in _ENCODERS:
        try:
            from sentence_transformers import SentenceTransformer
            _ENCODERS[key] = SentenceTransformer(EMBEDDING_MODEL, cache_folder=cache,
                                                device="cpu", revision=revision)
    return _ENCODERS[key]


def semantic_vectors(texts, encoder):
    tokenizer = encoder.tokenizer
    chunk_size = encoder.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    chunks, groups, weights = [], [], []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        indexes = []
        for start in range(0, len(tokens), chunk_size):
            part = tokens[start:start + chunk_size]
            indexes.append(len(chunks))
            chunks.append(tokenizer.decode(part, skip_special_tokens=True))
            weights.append(len(part))
        groups.append(indexes)
    embeddings = encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist()
    vectors = []
    for indexes in groups:
        total = sum(weights[i] for i in indexes)
        vectors.append([sum(embeddings[i][j] * weights[i] for i in indexes) / total
                        for j in range(len(embeddings[0]))])
    return vectors


def lexical_vectors(texts):
    counters = [Counter(re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е"))) for text in texts]
    words = sorted(set().union(*(counter.keys() for counter in counters)))
    return [[counter[word] for word in words] for counter in counters]


def duplication(texts, backend="semantic", encoder=None):"
    nonempty = [(i + 1, text.strip()) for i, text in enumerate(texts) if text.strip()]
    if len(nonempty) < 2:
        return {"duplication_score": None, "duplication_pairs": [], "pair_count": 0}
    values = [text for _, text in nonempty]
    if backend == "semantic":
        vectors = semantic_vectors(values, encoder or load_encoder())
    elif backend == "lexical":
        vectors = lexical_vectors(values)
    pairs = [{"agents": [nonempty[i][0], nonempty[j][0]], "similarity": cosine(vectors[i], vectors[j])}
             for i, j in combinations(range(len(vectors)), 2)]
    return {"duplication_score": sum(pair["similarity"] for pair in pairs) / len(pairs),
            "duplication_pairs": pairs, "pair_count": len(pairs)}


def evaluate(result, min_chars=100, backend="semantic", encoder=None):
    texts = [answer.get("text", "") for answer in result["finals"]]
    return {**coverage(result["sections"], min_chars),
            **duplication(texts, backend, encoder)}

"""Picking the passages of a paper that bear on a claim, and checking quoted evidence."""
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
VERDICTS = ("supports", "opposes", "irrelevant", "abstain")


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + size])
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def trim_to_sentences(chunk):
    first = chunk.find(". ")
    last = chunk.rfind(". ")
    if first == -1 or last == -1:
        return chunk
    start = first + 2
    end = last + 1
    if end <= start:
        return chunk
    trimmed = chunk[start:end]
    if len(trimmed) < 0.4 * len(chunk):
        return chunk
    return trimmed


def retrieve_chunks(claim, chunks, k=3):
    chunk_vecs = embed_model.encode(chunks)
    claim_vec = embed_model.encode(claim)
    scores = [cosine_similarity(claim_vec, cv) for cv in chunk_vecs]
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return ranked[:k]


def normalize(s):
    return " ".join(s.split())


def apply_quote_guard(verdict, quote, paper_text):
    """Never return evidence that isn't in the paper.

    supports/opposes must carry a quote found verbatim (whitespace-normalized)
    in the paper, or they are downgraded to abstain. irrelevant/abstain need no
    evidence, so their quote is dropped rather than checked — the model sometimes
    writes the verdict word itself into the quote field.
    Returns (verdict, quote, downgraded).
    """
    if verdict not in VERDICTS:
        return "abstain", "", False
    if verdict in ("irrelevant", "abstain"):
        return verdict, "", False
    if quote and normalize(quote) in normalize(paper_text):
        return verdict, quote, False
    return "abstain", "", True

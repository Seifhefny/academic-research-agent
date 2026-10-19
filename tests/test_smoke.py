import json

import pytest

from research_agent import check_claim, pipeline
from research_agent.evidence import apply_quote_guard, chunk_text, retrieve_chunks, trim_to_sentences
from research_agent.llm import parse_model_json
from research_agent.sources import pdf_bytes_to_text, search_arxiv, strip_references


def test_chunks_overlap_and_cover_text():
    text = "x" * 2500
    chunks = chunk_text(text, size=1000, overlap=200)
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0][-200:] == chunks[1][:200]
    assert "".join(c[:800] for c in chunks).startswith(text[:2400])


def test_strip_references_cuts_bibliography():
    assert strip_references("Body text.\nReferences\n[1] Foo") == "Body text.\n"
    assert strip_references("No bibliography here") == "No bibliography here"


def test_parse_model_json_handles_think_and_fences():
    raw = '<think>reasoning</think>\n```json\n{"verdict": "supports", "quote": "x"}\n```'
    assert parse_model_json(raw) == {"verdict": "supports", "quote": "x"}


def test_trim_to_sentences_drops_partial_edges():
    chunk = "tail of a sentence. Full sentence one. Full sentence two. head of next"
    assert trim_to_sentences(chunk) == "Full sentence one. Full sentence two."


def test_retrieve_chunks_finds_relevant_passage():
    chunks = ["The cat sat on the mat.", "Protein folding prediction with deep learning.", "Stock prices fell."]
    top = retrieve_chunks("AlphaFold predicts protein structures", chunks, k=1)
    assert top[0][1] == chunks[1]


PAPER = "Intro.  Predictions of alternative folds are often\ninaccurate. More text."


def test_guard_keeps_verbatim_quote_despite_whitespace():
    assert apply_quote_guard("supports", "alternative folds are often inaccurate.", PAPER) \
        == ("supports", "alternative folds are often inaccurate.", False)


def test_guard_downgrades_invented_quote():
    assert apply_quote_guard("opposes", "AlphaFold is always right.", PAPER) == ("abstain", "", True)
    assert apply_quote_guard("supports", "", PAPER) == ("abstain", "", True)


def test_guard_leaves_irrelevant_alone():
    # Seen in a real run: the model put the verdict word in the quote field.
    assert apply_quote_guard("irrelevant", "irrelevant", PAPER) == ("irrelevant", "", False)
    assert apply_quote_guard("banana", "x", PAPER) == ("abstain", "", False)


def test_malformed_pdf_returns_none_instead_of_crashing():
    assert pdf_bytes_to_text(b"%PDF-1.4 this is not really a pdf") is None


def test_pipeline_offline(monkeypatch):
    """Whole pipeline with network and LLM faked: one prefilter call, one bad PDF, one judged paper."""
    sources = [
        {"doi": f"id{i}", "title": f"Paper {i}", "authors": [], "year": "2024",
         "abstract": "abc", "pdf_url": f"pdf{i}"}
        for i in range(3)
    ]
    prefilter_calls = []
    monkeypatch.setattr(pipeline, "search_arxiv", lambda claim, rows: sources)
    monkeypatch.setattr(pipeline, "prefilter_by_title",
                        lambda claim, srcs: prefilter_calls.append(1) or srcs[:2])
    monkeypatch.setattr(pipeline, "fetch_pdf_text",
                        lambda url: None if url == "pdf1" else PAPER)
    monkeypatch.setattr(pipeline, "judge_claim", lambda claim, s, p: json.dumps(
        {"verdict": "supports", "quote": "alternative folds are often inaccurate.", "paper_question": "q"}))

    out = pipeline.check_claim("claim", rows=3)
    d = out["diagnostics"]
    assert len(prefilter_calls) == 1
    assert (d["sources_found"], d["kept_after_prefilter"], d["pdf_failures"], d["judged"]) == (3, 2, 1, 1)
    assert out["results"][0]["verdict"] == "supports"


@pytest.mark.live
def test_arxiv_search_returns_sources():
    sources = search_arxiv("protein structure prediction", rows=3)
    assert sources, "arXiv returned nothing (network or API change)"
    assert {"title", "abstract", "pdf_url"} <= sources[0].keys()


@pytest.mark.live
def test_check_claim_end_to_end(needs_ollama):
    out = check_claim("AlphaFold fails to predict alternative protein conformations", rows=3)
    d = out["diagnostics"]
    assert d["sources_found"] > 0
    assert d["judged"] == len(out["results"])
    for r in out["results"]:
        assert r["verdict"] in {"supports", "opposes", "irrelevant", "abstain"}

"""search -> prefilter -> read PDF -> retrieve passages -> judge -> guard."""
from .evidence import VERDICTS, apply_quote_guard, chunk_text, retrieve_chunks, trim_to_sentences
from .llm import judge_claim, parse_model_json, prefilter_by_title
from .sources import fetch_pdf_text, search_arxiv, strip_references


def check_claim(claim, rows=5):
    """Judge one claim against up to `rows` arXiv papers.

    Returns {"results": [...], "diagnostics": {...}}; each result carries a verdict
    and, for supports/opposes, a quote that is guaranteed to appear in the paper.
    """
    diagnostics = {
        "sources_found": 0,
        "kept_after_prefilter": 0,
        "pdf_failures": 0,
        "parse_failures": 0,
        "guard_downgrades": 0,
        "judged": 0,
        "verdict_counts": {v: 0 for v in VERDICTS},
    }

    sources = search_arxiv(claim, rows=rows)
    diagnostics["sources_found"] = len(sources)
    if not sources:
        return {"results": [], "diagnostics": diagnostics}

    kept = prefilter_by_title(claim, sources)
    diagnostics["kept_after_prefilter"] = len(kept)
    kept_dois = {s["doi"] for s in kept}
    for s in sources:
        print("KEEP " if s["doi"] in kept_dois else "DROP ", s["title"][:70])

    results = []
    for s in kept:
        text = fetch_pdf_text(s["pdf_url"])
        if text is None:
            diagnostics["pdf_failures"] += 1
            continue

        stripped = strip_references(text)
        chunks = chunk_text(stripped)
        top = retrieve_chunks(claim, chunks, k=3)
        passages = [trim_to_sentences(chunk) for score, chunk in top]

        try:
            parsed = parse_model_json(judge_claim(claim, s, passages))
        except Exception:
            diagnostics["parse_failures"] += 1
            continue

        raw_quote = parsed.get("quote") or ""
        verdict, quote, downgraded = apply_quote_guard(
            parsed.get("verdict", "abstain"), raw_quote, stripped
        )
        if downgraded:
            print("GUARD FIRED, quote was:", raw_quote[:150])
            diagnostics["guard_downgrades"] += 1

        results.append({
            "title": s["title"],
            "year": s["year"],
            "authors": s["authors"],
            "url": s["doi"],
            "verdict": verdict,
            "quote": quote,
            "paper_question": parsed.get("paper_question", ""),
        })
        diagnostics["judged"] += 1
        diagnostics["verdict_counts"][verdict] += 1

    return {"results": results, "diagnostics": diagnostics}

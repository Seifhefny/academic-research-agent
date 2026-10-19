"""The two LLM calls (relevance prefilter, evidence judge) via a local Ollama model."""
import json

import ollama

MODEL = "qwen3"


def prefilter_by_title(claim, sources):
    if not sources:
        return []
    listing = "\n\n".join(
    f"[{i}]\nTitle: {s['title']}\nAbstract: {s['abstract'][:300]}"
    for i, s in enumerate(sources,1)
    )
    
    prompt = f"""Claim: {claim} 


    Candidate sources:
    {listing}

For each source, decide whether the title AND the abstract suggests it is worth reading to evaluate this claim.
You are NOT deciding whether it supports or opposes the claim — titles cannot tell you that.
You are only deciding whether it is plausibly on-topic.


Reply with JSON only, no other text:
{{"keep": [1, 3]}}"""

    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    raw = r["message"]["content"]
    print("RAW PREFILTER:", raw[:500])
    parsed = parse_model_json(raw)
    print("PARSED:", parsed)
    keep_nums = parsed.get("keep", [])
    return [sources[n - 1] for n in keep_nums if 1 <= n <= len(sources)]


def judge_claim(claim, source, passages):
    evidence = "\n\n".join(passages)
    prompt = f"""Claim: {claim}

Passages from the paper "{source['title']}":

{evidence}

Does this paper support or oppose or abstain or irrelevant to the claim?

Rules:
- Base your answer ONLY on the passages above. Do not use outside knowledge.
- Quote one exact sentence from the passages that justifies your verdict (copy and paste DO NOT CHANGE A SINGLE LETTER).
- The quote must appear word-for-word in the passages.
- If the passages do not address the claim, use "irrelevant".
- If they address it but are not clear enough to decide, use "abstain" and leave quote empty.
- add a line in paper_question field stating what question the passages answer and return irrelevant if that question is not the one the claim makes

Reply with JSON only:
{{"paper_question": "...", "verdict": "supports|opposes|irrelevant|abstain", "quote": "..."}}"""

    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return r["message"]["content"]


def parse_model_json(raw):
    """Strip markdown fences and <think> blocks, then parse."""
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if "}" in text:
        text = text[:text.rfind("}") + 1]
    return json.loads(text)

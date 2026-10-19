"""Finding papers on arXiv and turning their PDFs into text."""
import io
import logging
import xml.etree.ElementTree as ET

import requests
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font/xref warnings flood the output

NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query, rows=5):
    params = {"search_query": f"all:{query}", "max_results": rows}
    try:
        r = requests.get("http://export.arxiv.org/api/query", params=params, timeout=30)
        root = ET.fromstring(r.text)
    except Exception as e:
        print("search failed:", type(e).__name__)
        print("BODY:", r.text[:200] if 'r' in dir() else "no response")
        return []

    sources = []
    for e in root.findall("atom:entry", NS):
        pdf = None
        for link in e.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf = link.get("href")
        sources.append({
            "doi": e.find("atom:id", NS).text,
            "title": e.find("atom:title", NS).text.strip(),
            "authors": [a.find("atom:name", NS).text for a in e.findall("atom:author", NS)],
            "year": e.find("atom:published", NS).text[:4],
            "abstract": e.find("atom:summary", NS).text.strip(),
            "pdf_url": pdf,
        })
    return sources


def fetch_pdf_text(pdf_url):
    """Download a PDF and extract its text. Returns None on failure."""
    try:
        r = requests.get(pdf_url, timeout=30)
    except Exception as e:
        print("  fetch failed:", type(e).__name__)

        return None

    if r.content[:5] != b"%PDF-":
        print("  not a PDF:", r.headers.get("Content-Type"))
        return None

    return pdf_bytes_to_text(r.content)


def pdf_bytes_to_text(data):
    """Extract text from PDF bytes. Returns None if the PDF can't be parsed."""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print("  pdf parse failed:", type(e).__name__)
        return None


def strip_references(text):
    cut = max(text.rfind("References"), text.rfind("REFERENCES"), text.rfind("Bibliography"))
    if cut == -1:
        return text
    return text[:cut]

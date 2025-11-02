# ingest_scientific.py — ArXiv + Wikidata ingestion for CLEU-AGI
import time, json, re, urllib.parse, feedparser
from typing import List, Tuple, Dict, Optional
from memory_store import add_chunks
from memory_graph import add_triple

UA = "CLEU-AGI/0.1 (+local)"

def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ---------------- ArXiv ----------------
def arxiv_search(query: str, max_results: int = 5) -> int:
    """
    Search ArXiv and add paper abstracts as chunks.
    Returns number of items added.
    """
    q = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
    feed = feedparser.parse(url)
    added = 0
    for e in feed.entries:
        title = _strip(e.title)
        summary = _strip(getattr(e, "summary", ""))
        link = ""
        for l in getattr(e, "links", []):
            if l.get("type") == "application/pdf":
                link = l.get("href", "")
        # add abstract as one chunk, include link in meta
        text = f"{title}\n\n{summary}\n\nPDF: {link}"
        add_chunks([text], {"url": link or getattr(e, "link", ""), "title": title, "source": "arxiv"})
        added += 1
    return added

# ---------------- Wikidata (very small helper) ----------------
# We don’t pull full SPARQL to avoid heavier deps; we accept users giving simple triples.
def wikidata_add(subject: str, predicate: str, obj: str) -> bool:
    """
    Minimal helper to insert a (subject, predicate, object) triple into CLEU graph memory.
    Use when you already know a fact you want to persist.
    """
    try:
        add_triple(subject, predicate, obj, source="wikidata:user")
        return True
    except Exception:
        return False

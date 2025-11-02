# ingest.py — Full-roam fetch/parse/chunk for CLEU-AGI with SAFE_MODE and LLM safety
import os, re, time, requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Tuple
from memory_store import add_chunks

# ============================================================
# CONFIG
# ============================================================
UA = "CLEU-AGI/1.0 (+local)"
REQUEST_TIMEOUT = 20
MAX_LEN = 1200            # Chunk size in characters
MAX_HOST_RPS = 30         # Max 30 requests/min per host
SAFE_MODE = False          # Controlled by /safe on|off

# LLM domain safety
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
SAFETY_TTL   = 6 * 3600  # Cache 6h

_safety_cache = {}  # host -> (timestamp, bool)
_host_times = {}    # host -> timestamps for rate limiting

# ============================================================
# UTILITIES
# ============================================================
def _rate(url: str):
    """Polite per-host rate limiting (max 30 requests/min)."""
    host = urlparse(url).netloc or ""
    now = time.time()
    bucket = _host_times.setdefault(host, [])
    _host_times[host] = [t for t in bucket if now - t < 60]
    if len(_host_times[host]) >= MAX_HOST_RPS:
        sleep_for = 60 - (now - _host_times[host][0])
        if sleep_for > 0:
            time.sleep(min(sleep_for, 2.0))
    _host_times[host].append(time.time())

def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _chunk(text: str, n: int = MAX_LEN) -> List[str]:
    """Split long text into roughly n-sized sentences."""
    s = _strip(text)
    if not s:
        return []
    if len(s) <= n:
        return [s]
    out, i = [], 0
    while i < len(s):
        j = min(len(s), i + n)
        win = s[i:j]
        m = re.search(r"[.!?]\s", win[-160:])
        if m:
            j = i + (len(win) - len(win[-160:]) + m.end())
            win = s[i:j]
        out.append(win.strip())
        i = j
    return out

# ============================================================
# LLM DOMAIN SAFETY
# ============================================================
def _llm_is_safe(host: str) -> bool:
    """Ask Ollama if the domain is generally safe to crawl."""
    now = time.time()
    rec = _safety_cache.get(host)
    if rec and (now - rec[0] < SAFETY_TTL):
        return rec[1]

    prompt = (
        "You are a URL safety classifier for an autonomous AI crawler.\n"
        "Input: a domain name.\n"
        "Decide if it's generally SAFE for automated, read-only access.\n"
        "Avoid unsafe, adult, malicious, or login-gated domains.\n"
        "Respond with one token: SAFE or UNSAFE.\n"
    )
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": f"{prompt}\n\nDomain: {host}\nAnswer:"}
            ],
            "stream": False,
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "message" in data and "content" in data["message"]:
            out = data["message"]["content"].strip().upper()
        elif "choices" in data and data["choices"]:
            out = data["choices"][0]["message"]["content"].strip().upper()
        else:
            out = "SAFE"
    except Exception:
        out = "SAFE"  # fail-open for resilience

    is_safe = ("SAFE" in out) and ("UNSAFE" not in out)
    _safety_cache[host] = (now, is_safe)
    return is_safe

def _allowed(url: str) -> bool:
    """Determine if a URL can be fetched."""
    if not SAFE_MODE:
        return True
    host = urlparse(url).netloc.lower()
    return _llm_is_safe(host)

# ============================================================
# FETCH + PARSE
# ============================================================
def fetch(url: str) -> Tuple[str, str, str]:
    """Fetch a URL safely, return (title, text, html)."""
    if not _allowed(url):
        raise RuntimeError(f"Domain not allowed in SAFE_MODE: {url}")
    _rate(url)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        bad.decompose()
    title = _strip(soup.title.get_text() if soup.title else url)
    text = _strip(soup.get_text(" "))
    return title, text, html

def links(base_url: str, html: str) -> List[str]:
    """Return deduped list of absolute links found in the page."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#"):
            continue
        full = urljoin(base_url, href)
        sch = urlparse(full).scheme
        if sch in ("http", "https") and (not SAFE_MODE or _allowed(full)):
            out.append(full)
    seen, deduped = set(), []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped

# ============================================================
# INGEST MAIN ENTRY
# ============================================================
def ingest_url(url: str, follow: int = 0) -> int:
    """
    Fetch `url`, chunk text, add to store, and optionally follow up to N links.
    Returns total number of pages/chunks processed.
    """
    title, text, html = fetch(url)
    chunks = _chunk(text)
    n = len(chunks)
    if n:
        add_chunks(chunks, {"url": url, "title": title, "source": "web"})

    taken = 0
    for lk in links(url, html):
        if taken >= follow:
            break
        try:
            t2, tx2, _ = fetch(lk)
            c2 = _chunk(tx2)
            add_chunks(c2, {"url": lk, "title": t2, "source": "web"})
            taken += 1
        except Exception:
            continue

    return n + taken

# ============================================================
# DEBUG TEST
# ============================================================
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Quantum_computing"
    print(f"[DEBUG] SAFE_MODE={SAFE_MODE}")
    try:
        total = ingest_url(test_url, follow=1)
        print(f"[DEBUG] Ingested {total} pages successfully.")
    except Exception as e:
        print(f"[ERROR] {e}")

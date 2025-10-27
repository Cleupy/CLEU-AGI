# ingest.py — fetch/parse/chunk with SAFE_MODE
import re, time, requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Tuple
from memory_store import add_chunks

SAFE_MODE = True
ALLOWED = {
    "wikipedia.org","nasa.gov","nih.gov","noaa.gov","data.gov",
    "arxiv.org","mit.edu","stanford.edu","loc.gov","whitehouse.gov"
}
UA = "CLEU-AGI/0.1 (+local)"
MAX_LEN = 1000

_host_times = {}
def _rate(url: str):
    host = urlparse(url).netloc
    now = time.time()
    _host_times.setdefault(host, [])
    _host_times[host] = [t for t in _host_times[host] if now - t < 60]
    if len(_host_times[host]) >= 30:
        time.sleep(60 - (now - _host_times[host][0]))
    _host_times[host].append(time.time())

def _allowed(url: str) -> bool:
    if not SAFE_MODE: return True
    host = urlparse(url).netloc.lower()
    return any(host.endswith(d) for d in ALLOWED)

def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _chunk(text: str, n: int = MAX_LEN) -> List[str]:
    s = _strip(text)
    if len(s) <= n: return [s] if s else []
    out, i = [], 0
    while i < len(s):
        j = min(len(s), i + n)
        win = s[i:j]
        m = re.search(r"[.!?]\s", win[-120:])
        if m:
            j = i + (len(win) - len(win[-120:]) + m.end())
            win = s[i:j]
        out.append(win.strip())
        i = j
    return out

def fetch(url: str) -> Tuple[str, str, str]:
    if SAFE_MODE and not _allowed(url):
        raise RuntimeError(f"Domain not allowed in SAFE_MODE: {url}")
    _rate(url)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script","style","nav","footer","header","noscript"]):
        bad.decompose()
    title = _strip(soup.title.get_text() if soup.title else url)
    text  = _strip(soup.get_text(" "))
    return title, text, html

def links(url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#"): continue
        full = urljoin(url, href)
        sch = urlparse(full).scheme
        if sch in ("http","https") and (not SAFE_MODE or _allowed(full)):
            out.append(full)
    # dedupe
    seen, deduped = set(), []
    for u in out:
        if u not in seen:
            deduped.append(u); seen.add(u)
    return deduped

def ingest_url(url: str, follow: int = 0) -> int:
    title, text, html = fetch(url)
    chunks = _chunk(text)
    n = add_chunks(chunks, {"url": url, "title": title, "source": "web"})
    taken = 0
    for lk in links(url, html):
        if taken >= follow: break
        try:
            t2, tx2, _ = fetch(lk)
            c2 = _chunk(tx2)
            add_chunks(c2, {"url": lk, "title": t2, "source": "web"})
            taken += 1
        except Exception:
            pass
    return n + taken

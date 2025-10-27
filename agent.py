# agent.py — CLEU-AGI reasoning, planning, reflection, autonomous loop
import os, time, json, traceback, random
from typing import List, Dict, Any

# == Optional Ollama (local) ==
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")

# == Try to use your ingest backend; fall back to a light local store ==
try:
    # expected from your ingest.py
    from ingest import search, add_text, ingest_url, SAFE_MODE, STORE_DIR
except Exception:
    # Fallback mini-store (only if ingest.py is missing or broken)
    import re, numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer

    SAFE_MODE = True
    STORE_DIR = "store"
    os.makedirs(STORE_DIR, exist_ok=True)
    EMB_PATH  = os.path.join(STORE_DIR, "embeddings.npy")
    META_PATH = os.path.join(STORE_DIR, "meta.jsonl")

    _hv = HashingVectorizer(n_features=4096, ngram_range=(1,2), alternate_sign=False, norm=None)

    def _load_embeddings():
        if os.path.exists(EMB_PATH):
            return np.load(EMB_PATH).astype("float32", copy=False)
        return np.zeros((0, 4096), dtype="float32")

    def _save_embeddings(E):
        np.save(EMB_PATH, E.astype("float32", copy=False))

    def _append_meta(m):
        with open(META_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def _load_meta():
        if not os.path.exists(META_PATH): return []
        out = []
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: out.append(json.loads(line))
                    except: pass
        return out

    def _encode(texts: List[str]):
        X = _hv.transform(texts)
        dense = X.toarray().astype("float32", copy=False)
        n = (dense**2).sum(axis=1, keepdims=True) ** 0.5
        n[n==0] = 1.0
        return dense / n

    def add_text(url: str, title: str, text: str, source: str="local") -> int:
        chunks = [text] if len(text) <= 1000 else [text[i:i+1000] for i in range(0, len(text), 1000)]
        if not chunks: return 0
        vecs = _encode(chunks)
        E = _load_embeddings()
        E = vecs if E.size == 0 else (np.vstack([E, vecs]))
        _save_embeddings(E)
        ts = time.time()
        for c in chunks:
            _append_meta({"url": url, "title": title, "snippet": c[:500], "source": source, "ts": ts})
        return len(chunks)

    def search(query: str, k: int = 6) -> List[Dict[str, Any]]:
        E = _load_embeddings()
        metas = _load_meta()
        if E.shape[0] == 0 or not metas: return []
        q = _encode([query])[0:1, :]
        sims = (E @ q.T).ravel()
        k = min(k, sims.size)
        idx = sims.argsort()[::-1][:k]
        out = []
        for i in idx:
            m = dict(metas[i]); m["score"] = float(sims[i]); out.append(m)
        return out

    def ingest_url(url: str, follow: int = 0) -> int:
        # Minimal fallback intake (does not follow links)
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(url, timeout=20, headers={"User-Agent":"CLEU-AGI/0.1"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            for bad in soup(["script","style","nav","footer","header","noscript"]):
                bad.decompose()
            title = soup.title.get_text(strip=True) if soup.title else url
            text = soup.get_text(" ", strip=True)
            return add_text(url, title, text, source="web")
        except Exception:
            return 0

# == Small local memory log ==
MEM_LOG = os.path.join(STORE_DIR, "memory_log.jsonl")

def _mem_write(kind: str, content: str, meta: Dict[str, Any] = None):
    rec = {"ts": time.time(), "kind": kind, "content": content, "meta": meta or {}}
    with open(MEM_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _mem_read(kind: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    if not os.path.exists(MEM_LOG): return []
    out = []
    with open(MEM_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if (kind is None) or (rec.get("kind") == kind):
                    out.append(rec)
            except:
                pass
    out.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return out[:limit]

# == Helpers ==
def _try_ollama(system: str, user: str, ctx_items: List[Dict[str, Any]]) -> str:
    import requests
    ctx = "\n".join([f"- {c['title']} ({c['url']}): {c['snippet']}" for c in (ctx_items or [])])
    prompt = (
        f"System:\n{system}\n\n"
        f"Context (use these facts; cite URLs inline):\n{ctx}\n\n"
        f"User:\n{user}\n\n"
        f"Assistant:\nConcise, factual answer with citations."
    )
    payload = {"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return "[no response]"

def _fallback_answer(question: str, ctx_items: List[Dict[str, Any]]) -> str:
    if not ctx_items:
        return "I don’t have enough local knowledge yet. Use /ingest <url> (and /safe off if you trust the domain)."
    lines = ["Here’s what I can recall locally:"]
    for c in ctx_items[:3]:
        lines.append(f"- {c['title']} ({c['url']}): {c['snippet'][:200]}…")
    lines.append("\n(No LLM reachable; answered from local context.)")
    return "\n".join(lines)

SYSTEM_POLICY = (
    "You are CLEU-AGI: careful, factual, concise. Prefer grounded citations. "
    "If unknown, say so and recommend safe next sources to learn from."
)

# == Public functions used by main.py ==
def answer(question: str) -> str:
    """RAG answer: retrieval -> (Ollama if available) -> log + store."""
    ctx = search(question, 6)
    try:
        out = _try_ollama(SYSTEM_POLICY, question, ctx)
    except Exception:
        out = _fallback_answer(question, ctx)

    # store as knowledge (short)
    add_text("local://answer", "Answer", out, source="answer")
    _mem_write("answer", out, {"q": question})
    return out

def plan(goal: str) -> str:
    """Make a tiny actionable plan; store it."""
    if not goal.strip():
        return "Provide a goal, e.g., /plan Learn cell biology basics in 2 weeks."
    try:
        plan_txt = _try_ollama(
            "Write a crisp 5-step plan with bullet points and concrete actions; 120 words max.",
            f"Goal: {goal}",
            search(goal, 4)
        )
    except Exception:
        plan_txt = (
            f"Plan for: {goal}\n"
            "- Define scope and materials\n- Schedule 30–60m blocks daily\n"
            "- Practice/review with spaced repetition\n- Do 1 mini-project\n- Review and adjust weekly"
        )
    add_text("local://plan", "Plan", plan_txt, source="plan")
    _mem_write("plan", plan_txt, {"goal": goal})
    return plan_txt

def reflect(text: str) -> str:
    """Self-critique a statement/answer; store it."""
    if not text.strip():
        return "Provide text to reflect on, e.g., /reflect My answer about X."
    try:
        refl = _try_ollama(
            "Critique for accuracy, missing assumptions, and safer next steps. 120 words max.",
            text,
            search(text, 4)
        )
    except Exception:
        refl = "Reflection: check sources, define assumptions, verify facts, and propose a quick test."
    add_text("local://reflection", "Reflection", refl, source="reflection")
    _mem_write("reflection", refl, {"about": text[:100]})
    return refl

def history(kind: str = None) -> str:
    """Return recent memory entries."""
    rows = _mem_read(kind)
    if not rows:
        return "(No memory yet.)"
    lines = []
    for r in rows:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
        lines.append(f"[{ts}] {r['kind']}: {r['content'][:200]}{'…' if len(r['content'])>200 else ''}")
    return "\n".join(lines)

# == Autonomous loop ==
SEEDS_SAFE = [
    "https://en.wikipedia.org/wiki/Artemis_program",
    "https://www.nasa.gov/",
    "https://www.nih.gov/news-events",
    "https://www.noaa.gov/",
    "https://www.loc.gov/",
    "https://www.whitehouse.gov/briefing-room/"
]

def autonomous_cycle():
    """
    One full cycle:
      1) Pick a seed (safe allowlisted if SAFE_MODE on).
      2) Ingest (+ a little follow).
      3) Ask a curiosity question based on recent content.
      4) Reflect on the answer.
    """
    # 1) Seed selection
    seed = random.choice(SEEDS_SAFE) if SAFE_MODE else random.choice([
        # when SAFE_MODE off, you could insert any seed list you trust
        "https://en.wikipedia.org/wiki/Systems_engineering",
        "https://en.wikipedia.org/wiki/Deep_reinforcement_learning",
    ])

    # 2) Ingest
    try:
        n = ingest_url(seed, follow=2)
        _mem_write("auto", f"Ingested from seed: {seed} (+{n})", {"seed": seed, "added": n})
        print(f"[AUTO] Ingested {n} chunks from {seed}")
    except Exception as e:
        _mem_write("auto", f"Ingest error: {e}", {"seed": seed})
        print(f"[AUTO] Ingest error: {e}")

    # 3) Ask a curiosity question
    curiosities = [
        "Summarize the most recent update and give 2 citations.",
        "What are open problems or next steps mentioned? Cite sources.",
        "List 3 key facts and 2 missing pieces of information.",
        "What is the historical context and why it matters now?"
    ]
    q = random.choice(curiosities)
    try:
        ans = answer(q)
        print(f"[AUTO] Q: {q}\n[AUTO] A: {ans}\n")
    except Exception as e:
        print(f"[AUTO] Answer error: {e}\n{traceback.format_exc()}")

    # 4) Reflect
    try:
        rf = reflect(ans if isinstance(ans, str) else json.dumps(ans))
        print(f"[AUTO] Reflection: {rf}\n")
    except Exception as e:
        print(f"[AUTO] Reflection error: {e}")

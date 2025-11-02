# agent.py — CLEU-AGI core: answer / plan / reflect / autonomous learn
# - Windows/Py3.14 friendly
# - No keyword args passed to add_chunks (fixes previous TypeError)
# - Confidence scoring + optional symbolic sanity check
# - Works even if SAFE_MODE isn't exported by ingest.py

from __future__ import annotations
import os, time, json, traceback, random, re
from typing import List, Dict, Any

# === LLM endpoint (Ollama local by default) ===
OLLAMA_URL  = os.environ.get("OLLAMA_URL",  "http://localhost:11434/api/chat")
OLLAMA_MODEL= os.environ.get("OLLAMA_MODEL","mistral:7b-instruct")

# === Core backends ===
from ingest import ingest_url  # SAFE_MODE may or may not exist — that's fine
try:
    from memory_store import search, add_chunks, STORE_DIR
except Exception:
    # graceful fallback if memory_store exports differ
    search = lambda q, k=6: []
    def add_chunks(chunks, meta=None): pass
    STORE_DIR = os.path.join(os.getcwd(), "store")

# Optional symbolic checker / graph (if present)
try:
    from reason_symbolic import check_consistency
except Exception:
    check_consistency = None

# === Local memory log ===
os.makedirs(STORE_DIR, exist_ok=True)
MEM_LOG = os.path.join(STORE_DIR, "memory_log.jsonl")

def _mem_write(kind: str, content: str, meta: Dict[str, Any] | None = None):
    rec = {"ts": time.time(), "kind": kind, "content": content, "meta": meta or {}}
    try:
        with open(MEM_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _mem_read(kind: str | None = None, limit: int = 20) -> List[Dict[str, Any]]:
    if not os.path.exists(MEM_LOG):
        return []
    out: List[Dict[str, Any]] = []
    with open(MEM_LOG, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if (kind is None) or (rec.get("kind") == kind):
                    out.append(rec)
            except Exception:
                pass
    out.sort(key=lambda r: r.get("ts", 0.0), reverse=True)
    return out[:limit]

# === LLM call ===
def _llm_chat(system: str, user: str, ctx_items: List[Dict[str, Any]]) -> str:
    import requests
    ctx = "\n".join(
        f"- {c.get('title','')} ({c.get('url','')}): {c.get('snippet','')}"
        for c in (ctx_items or [])
    )
    prompt = (
        f"System:\n{system}\n\n"
        f"Context (use these facts; cite URLs inline):\n{ctx}\n\n"
        f"User:\n{user}\n\nAssistant:"
    )
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return "[no response]"

def _fallback_answer(question: str, ctx_items: List[Dict[str, Any]]) -> str:
    if not ctx_items:
        return "No local context yet. Try /ingest <url> or /arxiv <query>."
    lines = ["Recalling from local memory:"]
    for c in ctx_items[:3]:
        lines.append(f"- {c.get('title','(no title)')} ({c.get('url','')}): {c.get('snippet','')[:200]}…")
    lines.append("\n(No LLM reachable; answered from local context.)")
    return "\n".join(lines)

SYSTEM_POLICY = (
    "You are CLEU-AGI: careful, factual, concise. Prefer grounded citations. "
    "If uncertain, say so and propose next reputable sources or tests."
)

# === Confidence scoring ===
_HEDGE = {"might","may","could","possibly","perhaps","appears","suggests","seems","likely","unclear"}
def _kw(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", (s or "").lower()))

def _count_citations(text: str) -> int:
    if not isinstance(text, str): return 0
    return len(re.findall(r"(https?://|doi:|\[\d+\]|\(\d{4}\))", text, flags=re.I))

def _overlap(a: str, b: str) -> float:
    A, B = _kw(a), _kw(b)
    if not A or not B: return 0.0
    inter = len(A & B)
    return inter / max(1, min(len(A), len(B)))

def _confidence(answer_text: str, ctx_items: List[Dict[str, Any]]) -> float:
    cits = min(4, _count_citations(answer_text))
    cit_score = (cits / 4.0) * 0.4
    ctx_score = 0.0
    if ctx_items:
        overlaps = [_overlap(answer_text, c.get("snippet","")) for c in ctx_items[:6]]
        ctx_score = (sum(overlaps)/max(1,len(overlaps))) * 0.4
    hedges = len(_kw(answer_text) & _HEDGE)
    hedge_pen = -min(0.2, hedges * 0.04)
    return float(max(0.0, min(1.0, cit_score + ctx_score + hedge_pen)))

# === Public API ===
def answer(question: str) -> str:
    ctx = search(question, 6) if callable(search) else []
    try:
        out = _llm_chat(SYSTEM_POLICY, question, ctx)
    except Exception:
        out = _fallback_answer(question, ctx)
    # persist as a single chunk
    add_chunks([out], {"url": "local://answer", "title": "Answer", "kind": "answer"})
    _mem_write("answer", out, {"q": question})
    return out

def plan(goal: str) -> str:
    if not goal.strip():
        return "Provide a goal, e.g., /plan Learn linear algebra in 2 weeks."
    try:
        out = _llm_chat(
            "Write a concise 5-step plan with concrete daily actions; 120 words max.",
            f"Goal: {goal}",
            search(goal, 4) if callable(search) else []
        )
    except Exception:
        out = (
            f"Plan for: {goal}\n"
            "- Define scope/materials\n- Study daily 1 hour\n"
            "- Practice applied examples\n- Build a mini-project\n- Review weekly"
        )
    add_chunks([out], {"url":"local://plan","title":"Plan","kind":"plan"})
    _mem_write("plan", out, {"goal": goal})
    return out

def reflect(text: str) -> str:
    if not text.strip():
        return "Provide text to reflect on, e.g., /reflect My answer about X."
    ctx = search(text, 6) if callable(search) else []
    try:
        out = _llm_chat(
            "Critique for accuracy, assumptions, and safer next steps. 120 words max.",
            text,
            ctx
        )
    except Exception:
        out = "Reflection: verify math rigor, double-check sources, and test small examples."
    conf = _confidence(out, ctx)
    # optional symbolic sanity
    try:
        if check_consistency:
            ok = check_consistency(out)
            out = f"{out}\n\n[logic:{'OK' if ok else 'CHECK'}]"
    except Exception:
        pass
    add_chunks([out], {"url":"local://reflection","title":"Reflection","kind":"reflection","confidence": conf})
    _mem_write("reflection", out, {"about": text[:160], "confidence": conf})
    return f"{out}\n\n[confidence: {conf:.2f}]"

def history(kind: str | None = None) -> str:
    rows = _mem_read(kind)
    if not rows: return "(No memory yet.)"
    lines = []
    for r in rows:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("ts",0)))
        body = r.get("content","")
        lines.append(f"[{ts}] {r.get('kind','?')}: {body[:200]}{'…' if len(body)>200 else ''}")
    return "\n".join(lines)

# === Autonomous learning loop (non-blocking driver is in main.py) ===
SEEDS = [
    "https://en.wikipedia.org/wiki/Algorithm",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Information_theory",
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://arxiv.org/list/cs.AI/recent",
    "https://arxiv.org/list/quant-ph/recent",
]

def autonomous_cycle():
    """One full cycle: ingest → ask → reflect."""
    seed = random.choice(SEEDS)
    try:
        n = ingest_url(seed, follow=2)
        _mem_write("auto", f"Ingested from seed: {seed} (+{n})", {"seed": seed, "added": n})
        print(f"[AUTO] Ingested {n} from {seed}")
    except Exception as e:
        _mem_write("auto", f"Ingest error: {e}", {"seed": seed})
        print(f"[AUTO] Ingest error: {e}")

    curiosities = [
        "Summarize the most recent update and cite two URLs.",
        "List three key concepts and two unanswered questions.",
        "Explain a mathematical principle behind this topic; cite one source.",
        "Relate this to computational efficiency or complexity.",
        "Describe a practical application and cite a reputable paper.",
    ]
    q = random.choice(curiosities)
    try:
        ans = answer(q)
        print(f"[AUTO] Q: {q}\n[AUTO] A: {ans[:300]}{'…' if len(ans)>300 else ''}\n")
    except Exception as e:
        print(f"[AUTO] Answer error: {e}\n{traceback.format_exc()}")

    try:
        rf = reflect(ans if isinstance(ans, str) else json.dumps(ans))
        print(f"[AUTO] Reflection: {rf[:300]}{'…' if len(rf)>300 else ''}\n")
    except Exception as e:
        print(f"[AUTO] Reflection error: {e}")

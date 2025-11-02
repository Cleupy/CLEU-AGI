# agent.py — CLEU-AGI reasoning, planning, reflection, autonomous loop (clean + full roam)
import os, time, json, traceback, random
from typing import List, Dict, Any

# === LLM connection (Ollama optional) ===
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")

# === Import core backends ===
from ingest import ingest_url  # roaming handled in ingest.py
from memory_store import search, add_chunks, STORE_DIR

# --- keep SAFE_MODE control entirely inside ingest.py via /safe on|off in main (if desired) ---

# === Local memory log ===
MEM_LOG = os.path.join(STORE_DIR, "memory_log.jsonl")

def _mem_write(kind: str, content: str, meta: Dict[str, Any] = None):
    rec = {"ts": time.time(), "kind": kind, "content": content, "meta": meta or {}}
    os.makedirs(STORE_DIR, exist_ok=True)
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

# === Small shim so older add_text(...) calls still work ===
def add_text(url: str, title: str, text: str, source: str = "local") -> int:
    """Adapters old signature -> memory_store.add_chunks(chunks, meta)."""
    MAX_CHUNK = 1000
    if not isinstance(text, str): text = str(text)
    chunks = [text] if len(text) <= MAX_CHUNK else [text[i:i+MAX_CHUNK] for i in range(0, len(text), MAX_CHUNK)]
    meta = {"url": url, "title": title, "source": source}
    return add_chunks(chunks, meta)

# === Helper: call Ollama or fallback ===
def _try_ollama(system: str, user: str, ctx_items: List[Dict[str, Any]]) -> str:
    import requests
    ctx = "\n".join([f"- {c.get('title','?')} ({c.get('url','?')}): {c.get('snippet','')}" for c in (ctx_items or [])])
    prompt = (
        f"System:\n{system}\n\n"
        f"Context (use these facts; cite URLs inline):\n{ctx}\n\n"
        f"User:\n{user}\n\nAssistant:"
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
        return "I don’t have enough local knowledge yet. Use /ingest <url> or keep the autonomous loop running."
    lines = ["Here’s what I can recall locally:"]
    for c in ctx_items[:3]:
        title = c.get("title","?")
        url   = c.get("url","?")
        snip  = c.get("snippet","")[:200]
        lines.append(f"- {title} ({url}): {snip}…")
    lines.append("\n(No LLM reachable; answered from local context.)")
    return "\n".join(lines)

SYSTEM_POLICY = (
    "You are CLEU-AGI: careful, factual, concise. Prefer grounded citations. "
    "If unknown, say so and suggest reputable next sources to learn from."
)

# === Public API for main.py ===
def answer(question: str) -> str:
    ctx = search(question, 6)
    try:
        out = _try_ollama(SYSTEM_POLICY, question, ctx)
    except Exception:
        out = _fallback_answer(question, ctx)
    add_text("local://answer", "Answer", out, source="answer")
    _mem_write("answer", out, {"q": question})
    return out

def plan(goal: str) -> str:
    if not goal.strip():
        return "Provide a goal, e.g., /plan Learn linear algebra in 2 weeks."
    try:
        plan_txt = _try_ollama(
            "Write a concise 5-step plan with concrete daily actions; 120 words max.",
            f"Goal: {goal}",
            search(goal, 4)
        )
    except Exception:
        plan_txt = (
            f"Plan for: {goal}\n"
            "- Define scope and materials\n- Study daily 1 hour\n"
            "- Practice with applied examples\n- Build a mini-project\n- Review progress weekly"
        )
    add_text("local://plan", "Plan", plan_txt, source="plan")
    _mem_write("plan", plan_txt, {"goal": goal})
    return plan_txt

def reflect(text: str) -> str:
    if not text.strip():
        return "Provide text to reflect on, e.g., /reflect My answer about X."
    try:
        refl = _try_ollama(
            "Critique for accuracy, missing assumptions, and safer next steps. 120 words max.",
            text,
            search(text, 4)
        )
    except Exception:
        refl = "Reflection: verify math rigor, double-check sources, and test small examples."
    add_text("local://reflection", "Reflection", refl, source="reflection")
    _mem_write("reflection", refl, {"about": text[:100]})
    return refl

def history(kind: str = None) -> str:
    rows = _mem_read(kind)
    if not rows:
        return "(No memory yet.)"
    lines = []
    for r in rows:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
        lines.append(f"[{ts}] {r['kind']}: {r['content'][:200]}{'…' if len(r['content'])>200 else ''}")
    return "\n".join(lines)

# === Autonomous loop (full roam; broad seeds) ===
SEEDS = [
    # Core CS
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Algorithm",
    "https://en.wikipedia.org/wiki/Data_structure",
    "https://en.wikipedia.org/wiki/Time_complexity",
    "https://en.wikipedia.org/wiki/Computational_complexity_theory",
    "https://en.wikipedia.org/wiki/Operating_system",
    "https://en.wikipedia.org/wiki/Compiler",
    "https://en.wikipedia.org/wiki/Database",
    "https://en.wikipedia.org/wiki/Computer_network",
    "https://en.wikipedia.org/wiki/Cryptography",
    # Math foundations
    "https://en.wikipedia.org/wiki/Linear_algebra",
    "https://en.wikipedia.org/wiki/Calculus",
    "https://en.wikipedia.org/wiki/Probability_theory",
    "https://en.wikipedia.org/wiki/Statistics",
    "https://en.wikipedia.org/wiki/Information_theory",
    # AI/ML
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://arxiv.org/list/cs.LG/recent",
    "https://arxiv.org/list/cs.AI/recent",
    # Quantum
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://en.wikipedia.org/wiki/Quantum_algorithm",
    "https://en.wikipedia.org/wiki/Quantum_Fourier_transform",
    "https://en.wikipedia.org/wiki/Grover%27s_algorithm",
    "https://en.wikipedia.org/wiki/Shor%27s_algorithm",
    "https://en.wikipedia.org/wiki/Quantum_error_correction",
    "https://arxiv.org/list/quant-ph/recent",
]

def autonomous_cycle():
    """
    One full self-learning cycle:
      1) Choose a seed
      2) Ingest + follow related links (broad roam handled by ingest.py)
      3) Ask a curiosity question based on recent content
      4) Reflect on the answer
    """
    seed = random.choice(SEEDS)
    try:
        # roam more aggressively each cycle
        n = ingest_url(seed, follow=8)
        _mem_write("auto", f"Ingested from seed: {seed} (+{n})", {"seed": seed, "added": n})
        print(f"[AUTO] Ingested {n} items from {seed}")
    except Exception as e:
        _mem_write("auto", f"Ingest error: {e}", {"seed": seed})
        print(f"[AUTO] Ingest error: {e}")

    curiosities = [
        "Summarize the most recent update and cite two URLs.",
        "List three key concepts and two unanswered research questions.",
        "Explain a mathematical principle behind this topic; cite one source.",
        "How does this relate to computational efficiency or complexity?",
        "Describe a practical quantum or computational application; cite one reputable paper.",
    ]
    q = random.choice(curiosities)
    try:
        ans = answer(q)
        print(f"[AUTO] Q: {q}\n[AUTO] A: {ans}\n")
    except Exception as e:
        print(f"[AUTO] Answer error: {e}\n{traceback.format_exc()}")
        ans = ""

    try:
        rf = reflect(ans if isinstance(ans, str) else json.dumps(ans))
        print(f"[AUTO] Reflection: {rf}\n")
    except Exception as e:
        print(f"[AUTO] Reflection error: {e}")


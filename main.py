# main.py — CLEU-AGI interactive console + non-blocking autonomous runner
# - Smooth Windows console UX, transcript logging
# - Adds /arxiv, /fact, /graph show|path, /recall, /status, /version
# - SAFE_MODE toggle if ingest exposes it
# - "end of line" on exit

from __future__ import annotations
import os, shlex, time, traceback, threading, datetime, re

from agent import answer, plan, reflect, history, autonomous_cycle
from ingest import ingest_url
import ingest  # for SAFE_MODE if available

try:
    from memory_store import search as recall_search, STORE_DIR
except Exception:
    recall_search, STORE_DIR = None, os.path.join(os.getcwd(), "store")

# ---------- Banner ----------
BANNER = r"""
╔══════════════════════════════════════════════╗
║ CLEU-AGI: Codified Likeness Evolving Utility ║
║  Mission: I will create the perfect system   ║
╚══════════════════════════════════════════════╝
Type /help for commands. /exit to quit.
"""

# ---------- Transcript ----------
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
SESSION_LOG = os.path.join(LOG_DIR, f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def _log(line: str):
    try:
        with open(SESSION_LOG, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass

def _p(s: str): print(s); _log(s)
def _err(s: str): print(s); _log(s)
def _ok(s: str): print(s); _log(s)

# ---------- SAFE_MODE helpers ----------
def _safe_supported() -> bool:
    return hasattr(ingest, "SAFE_MODE")

def _safe_get():
    return getattr(ingest, "SAFE_MODE", None)

def _safe_set(val: bool) -> bool:
    if _safe_supported():
        setattr(ingest, "SAFE_MODE", bool(val))
        return True
    return False

# ---------- Autonomous runner (threaded) ----------
_auto_thread: threading.Thread | None = None
_auto_stop = threading.Event()

def _auto_loop(interval_min: int):
    _ok(f"[AUTO] CLEU-AGI autonomous mode engaged — every {interval_min} minutes.")
    try:
        while not _auto_stop.is_set():
            try:
                autonomous_cycle()
            except Exception as e:
                _err(f"[AUTO ERROR] {e}")
                traceback.print_exc()
            total = max(1, interval_min) * 60
            for _ in range(total):
                if _auto_stop.is_set(): break
                time.sleep(1)
    finally:
        _ok("[AUTO] Autonomous mode stopped.")

def start_auto(interval_min: int = 30):
    global _auto_thread
    if _auto_thread and _auto_thread.is_alive():
        _ok("[AUTO] Already running.")
        return
    _auto_stop.clear()
    _auto_thread = threading.Thread(target=_auto_loop, args=(interval_min,), daemon=True)
    _auto_thread.start()

def stop_auto():
    if _auto_thread and _auto_thread.is_alive():
        _ok("[AUTO] Stopping autonomous loop…")
        _auto_stop.set()
        _auto_thread.join(timeout=5)
    else:
        _ok("[AUTO] No autonomous loop active.")

# ---------- Utilities ----------
_URL_RE = re.compile(r"^https?://", re.I)

def _parse_ingest_args(args: list[str]) -> tuple[str | None, int]:
    url, follow, i = None, 0, 0
    while i < len(args):
        tok = args[i]
        if tok == "--follow" and (i + 1) < len(args):
            try:
                follow = int(args[i + 1])
            except Exception:
                _err("[Parse Error] --follow expects an integer.")
            i += 2
            continue
        if url is None:
            url = tok
        i += 1
    return url, follow

def _validate_url(u: str) -> bool: return bool(_URL_RE.search(u or ""))

def _human_size(path: str) -> str:
    try:
        n = os.path.getsize(path)
        for unit in ("B","KB","MB","GB"):
            if n < 1024: return f"{n:.0f}{unit}"
            n /= 1024
        return f"{n:.1f}TB"
    except Exception:
        return "?"

def _status():
    emb = os.path.join(STORE_DIR or "store", "embeddings.npy")
    meta = os.path.join(STORE_DIR or "store", "meta.jsonl")
    meml= os.path.join(STORE_DIR or "store", "memory_log.jsonl")
    lines = [
        "[STATUS]",
        f"  STORE_DIR: {STORE_DIR}",
        f"  embeddings.npy: {'present' if os.path.exists(emb) else 'missing'} ({_human_size(emb)})",
        f"  meta.jsonl:     {'present' if os.path.exists(meta) else 'missing'} ({_human_size(meta)})",
        f"  memory_log:     {'present' if os.path.exists(meml) else 'missing'} ({_human_size(meml)})",
        f"  SAFE_MODE:      {('on' if _safe_get() else 'off') if _safe_supported() else 'unsupported'}",
        f"  Transcript:     {SESSION_LOG}",
    ]
    return "\n".join(lines)

def _versions():
    import platform
    return "\n".join([
        "[VERSION]",
        f"  Python: {platform.python_version()}",
        f"  Platform: {platform.platform()}",
        f"  OLLAMA_URL: {os.environ.get('OLLAMA_URL','http://localhost:11434/api/chat')}",
        f"  OLLAMA_MODEL: {os.environ.get('OLLAMA_MODEL','mistral:7b-instruct')}",
    ])

def _show_help():
    _p("""
Commands
  /help                          Show this help
  /exit                          Quit (prints "end of line")
  /goodbye                       Alias for /exit
  /ask <question>                Ask a factual or reasoning question
  /plan <goal>                   Create a short goal plan
  /reflect <text>                Self-evaluate a statement or idea
  /mem [kind]                    Show stored memories (answer/plan/reflection)
  /recall <query>                Vector recall top-5 (title/url/score/snippet)
  /ingest <url> [--follow N]     Fetch page and index (+ follow N internal links)
  /safe on|off                   Toggle SAFE_MODE (if supported by ingest.py)
  /arxiv <query> [N]             Ingest N (default 5) ArXiv abstracts
  /fact S | p | O                Add a triple to the graph memory
  /graph show <entity>           Show in/out triples for entity
  /graph path A | B              Show a short path A→…→B (≤3 hops)
  /auto [minutes]                Start autonomous loop (non-blocking)
  /stopauto                      Stop autonomous loop if running
  /status                        Show store + runtime status
  /version                       Show Python/platform/model info
  /config                        Show core config paths
  /cls                           Clear screen
""")

def _config():
    return "\n".join([
        "[CONFIG]",
        f"  CWD: {os.getcwd()}",
        f"  STORE_DIR: {STORE_DIR}",
        f"  LOG_DIR: {LOG_DIR}",
        f"  SESSION_LOG: {SESSION_LOG}",
    ])

def _clear(): os.system("cls" if os.name == "nt" else "clear")

# ---------- Main loop ----------
def main():
    _p(BANNER)
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _p("end of line")
            stop_auto()
            break

        if not line:
            continue

        _log("You: " + line)

        if not line.startswith("/"):
            try:
                _p(answer(line))
            except Exception as e:
                _err(f"[Error] {e}")
            continue

        # Commands
        try:
            parts = shlex.split(line)
        except Exception:
            _err("[Parse Error] Invalid command format.")
            continue

        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("/exit", "/quit", "/end", "/goodbye"):
            stop_auto()
            _p("end of line")
            return

        elif cmd == "/help":
            _show_help()

        elif cmd == "/cls":
            _clear()

        elif cmd == "/ask":
            _p(answer(" ".join(args)))

        elif cmd == "/plan":
            _p(plan(" ".join(args)))

        elif cmd == "/reflect":
            _p(reflect(" ".join(args)))

        elif cmd == "/mem":
            _p(history(args[0] if args else None))

        elif cmd == "/recall":
            if recall_search is None:
                _err("[Recall] memory_store.search not available.")
                continue
            if not args:
                _err("Usage: /recall <query>")
                continue
            hits = recall_search(" ".join(args), 5)
            if not hits:
                _p("[Recall] No matches.")
            else:
                lines = ["[Recall Top-5]"]
                for i, h in enumerate(hits, 1):
                    lines.append(
                        f"{i}. {h.get('title','(no title)')}  "
                        f"[{h.get('score',0):.3f}]\n   {h.get('url','')}\n   {h.get('snippet','')[:180]}{'…' if len(h.get('snippet',''))>180 else ''}"
                    )
                _p("\n".join(lines))

        elif cmd == "/ingest":
            url, follow = _parse_ingest_args(args)
            if not url:
                _err("Usage: /ingest <url> [--follow N]"); continue
            if not _validate_url(url):
                _err("[Ingest] URL must start with http:// or https://"); continue
            try:
                n = ingest_url(url, follow)
                _ok(f"[Ingested] {n} items.")
            except Exception as e:
                _err(f"[Ingest Failed] {e}")

        elif cmd == "/safe":
            if not args:
                if _safe_supported():
                    _p(f"SAFE_MODE is {'on' if _safe_get() else 'off'}")
                else:
                    _p("SAFE_MODE not supported by current ingest.py")
            else:
                val = args[0].lower() in ("on","true","1","yes")
                if _safe_set(val):
                    _ok(f"SAFE_MODE → {'on' if val else 'off'}")
                else:
                    _p("SAFE_MODE not supported by current ingest.py")

        elif cmd == "/arxiv":
            if not args:
                _err("Usage: /arxiv <query> [max_results]"); continue
            try:
                from ingest_scientific import arxiv_search
            except Exception:
                _err("[ArXiv] Module not found. Add ingest_scientific.py."); continue
            q = " ".join(args[:-1]) if args[-1].isdigit() else " ".join(args)
            n = int(args[-1]) if args and args[-1].isdigit() else 5
            try:
                added = arxiv_search(q, n)
                _ok(f"[ArXiv] Added {added} abstracts for '{q}'.")
            except Exception as e:
                _err(f"[ArXiv] Failed: {e}")

        elif cmd == "/fact":
            raw = " ".join(args)
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) != 3:
                _err("Usage: /fact Subject | predicate | Object"); continue
            try:
                from memory_graph import add_triple
                add_triple(parts[0], parts[1], parts[2], source="user")
                _ok(f"[Graph] (+) ({parts[0]}) -[{parts[1]}]-> ({parts[2]})")
            except Exception as e:
                _err(f"[Graph] Failed: {e}")

        elif cmd == "/graph":
            if not args:
                _err("Usage: /graph show <entity>  OR  /graph path A | B"); continue
            sub = args[0].lower()
            if sub == "show":
                ent = " ".join(args[1:])
                try:
                    from memory_graph import triples_about
                    rows = triples_about(ent, direction="both")
                    if not rows: _p("[Graph] (no triples)")
                    else:
                        _p("[Graph] triples:")
                        for r in rows:
                            _p(f"  ({r['s']}) -[{r['p']}]-> ({r['o']})   [{r.get('source','')}]")
                except Exception as e:
                    _err(f"[Graph] Failed: {e}")
            elif sub == "path":
                raw = " ".join(args[1:])
                parts = [p.strip() for p in raw.split("|")]
                if len(parts) != 2:
                    _err("Usage: /graph path A | B"); continue
                try:
                    from memory_graph import find_path
                    path = find_path(parts[0], parts[1], max_len=3)
                    _p(f"[Graph] path: { ' -> '.join(path) if path else '(none)' }")
                except Exception as e:
                    _err(f"[Graph] Failed: {e}")
            else:
                _err("Usage: /graph show <entity>  OR  /graph path A | B")

        elif cmd == "/auto":
            try:
                interval = int(args[0]) if args else 30
            except Exception:
                interval = 30
            start_auto(interval)

        elif cmd == "/stopauto":
            stop_auto()

        elif cmd == "/status":
            _p(_status())

        elif cmd == "/version":
            _p(_versions())

        elif cmd == "/config":
            _p("\n".join([ "[CONFIG]",
                f"  CWD: {os.getcwd()}",
                f"  STORE_DIR: {STORE_DIR}",
                f"  LOG_DIR: {LOG_DIR}",
                f"  SESSION_LOG: {SESSION_LOG}",
            ]))

        else:
            _p("[Unknown] Type /help for available commands.")

if __name__ == "__main__":
    main()

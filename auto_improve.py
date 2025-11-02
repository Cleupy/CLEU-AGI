# auto_improve.py — self-coding loop (memory-aware, robust, Windows-friendly)
# Key features:
# - Timezone-aware datetimes (UTC)
# - Rotating logs + console echo
# - Exponential backoff + retry for Ollama
# - Diff extraction (prose or ```diff fenced) with strict validation
# - Git dry-run before apply; snapshot fallback if no git
# - Strict allow/deny and basename/path-traversal guards
# - Reads recent code, proposals, and agent memories for context
# - CLI flags: --once, --goal, --interval, --only-on-new
# - No external dependencies beyond stdlib + requests

import os
import re
import sys
import json
import time
import shutil
import pathlib
import logging
from logging.handlers import RotatingFileHandler
import subprocess
from typing import List, Optional
from datetime import datetime, UTC

import requests

from auto_config import (
    MODEL_NAME, OLLAMA_URL, CYCLE_SECONDS,
    ALLOWED_EDIT, BLOCKLIST, MAX_PATCH_CHARS,
    RUN_TESTS, USE_GIT, STATIC_CHECK,
)

# Optional toggles
ONLY_ON_NEW_MEMORIES = getattr(__import__("auto_config"), "ONLY_ON_NEW_MEMORIES", False)
MEMORY_DELTA_WINDOW  = getattr(__import__("auto_config"), "MEMORY_DELTA_WINDOW", 200)

ROOT       = pathlib.Path(__file__).resolve().parent
LOGS       = ROOT / "auto_logs"
PROPOSALS  = ROOT / "proposals"
STORE      = ROOT / "store"
LOGS.mkdir(exist_ok=True)
PROPOSALS.mkdir(exist_ok=True)

LAST_MEM_TS_FILE = LOGS / "last_memory_ts.txt"

# ----------------------------- Logging ---------------------------------
LOG_PATH = LOGS / "auto_improve.log"

_logger = logging.getLogger("auto_improve")
_logger.setLevel(logging.INFO)
# rotating file
_fh = RotatingFileHandler(str(LOG_PATH), maxBytes=1024 * 1024, backupCount=3)
_fh.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
# console
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_fh)
_logger.addHandler(_ch)

def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def _status(msg: str):
    _logger.info(msg)

# ---------------------------- Shell helpers ----------------------------
def _run(cmd: List[str], cwd: pathlib.Path = ROOT) -> subprocess.CompletedProcess:
    # shell=False critical for Windows safety; capture stdout/stderr
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, shell=False
    )

def git_available() -> bool:
    if not USE_GIT:
        return False
    try:
        r = _run(["git", "--version"])
        return r.returncode == 0
    except Exception:
        return False

def git_init_if_needed():
    if not git_available():
        return
    if not (ROOT / ".git").exists():
        _run(["git", "init"])
        _run(["git", "add", "."])
        _run(["git", "commit", "-m", "auto-improve: baseline"])

# ------------------------ Snapshot (no-git fallback) -------------------
def snapshot_backup() -> pathlib.Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snap = ROOT / f"_backup_{ts}"
    snap.mkdir(exist_ok=False)
    for p in ROOT.iterdir():
        if p.name.startswith("_backup_") or p.name in {".git", "proposals", "auto_logs"}:
            continue
        dst = snap / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)
    return snap

def restore_snapshot(snap: pathlib.Path):
    for p in list(ROOT.iterdir()):
        if p.name in (".git", snap.name) or p.name.startswith("_backup_"):
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except Exception:
            pass
    for p in snap.iterdir():
        dst = ROOT / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)
    shutil.rmtree(snap, ignore_errors=True)

# --------------------------- File utilities ----------------------------
def _read_text(p: pathlib.Path, limit: int) -> str:
    try:
        return p.read_text(encoding="utf-8")[:limit]
    except Exception:
        return ""

def _tail_lines(path: pathlib.Path, n: int) -> List[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:] if n and n > 0 else lines
    except Exception:
        return []

# --------------------------- Memory helpers ----------------------------
def _latest_memory_ts(window: int = MEMORY_DELTA_WINDOW) -> float:
    memlog = STORE / "memory_log.jsonl"
    latest = 0.0
    for ln in _tail_lines(memlog, window):
        try:
            j = json.loads(ln)
            ts = float(j.get("ts", 0.0) or 0.0)
            if ts > latest:
                latest = ts
        except Exception:
            pass
    return latest

def _load_last_applied_ts() -> float:
    if LAST_MEM_TS_FILE.exists():
        try:
            return float(LAST_MEM_TS_FILE.read_text(encoding="utf-8").strip() or 0.0)
        except Exception:
            return 0.0
    return 0.0

def _store_last_applied_ts(ts: float):
    try:
        LAST_MEM_TS_FILE.write_text(str(ts), encoding="utf-8")
    except Exception:
        pass

# ----------------------- Context builders for LLM ----------------------
def _collect_code_context() -> str:
    items = []
    for fname in sorted(ALLOWED_EDIT):
        p = ROOT / fname
        if p.exists() and p.is_file():
            txt = _read_text(p, 8000)
            if txt:
                items.append(f"== {fname} ==\n{txt}")
    return "\n\n".join(items)

def _collect_recent_proposals(k: int = 3) -> str:
    paths = sorted(PROPOSALS.glob("*.json"), key=lambda x: x.stat().st_mtime)[-k:]
    out = []
    for pj in paths:
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
            out.append(f"== PROPOSAL {pj.name} ==\n{json.dumps(data, ensure_ascii=False)}")
        except Exception:
            pass
    return "\n\n".join(out)

def _collect_agent_memories(k: int = 12) -> str:
    chunks = []
    # memory_log.jsonl
    memlog = STORE / "memory_log.jsonl"
    if memlog.exists():
        short = []
        for ln in [ln for ln in _tail_lines(memlog, k) if ln.strip()]:
            try:
                rec = json.loads(ln)
                kind = rec.get("kind", "")
                content = rec.get("content", "")
                meta = rec.get("meta", {})
                ts = rec.get("ts", 0)
                snippet = (content if isinstance(content, str) else json.dumps(content))[:400]
                short.append({"ts": ts, "kind": kind, "snippet": snippet, "meta": meta})
            except Exception:
                pass
        if short:
            chunks.append("== RECENT_MEMORIES ==\n" + json.dumps(short, ensure_ascii=False))
    # meta.jsonl (recent ingest titles/urls)
    meta = STORE / "meta.jsonl"
    if meta.exists():
        items = []
        for ln in _tail_lines(meta, k):
            try:
                j = json.loads(ln)
                items.append({"title": j.get("title", ""), "url": j.get("url", "")})
            except Exception:
                pass
        if items:
            chunks.append("== RECENT_INGEST ==\n" + json.dumps(items, ensure_ascii=False))
    return "\n\n".join(chunks)

def collect_context() -> str:
    parts = [
        _collect_code_context(),
        _collect_recent_proposals(),
        _collect_agent_memories()
    ]
    return "\n\n".join([p for p in parts if p])

# -------------------------- LLM interaction ---------------------------
SYSTEM = (
    "You are a precise software agent that proposes SMALL, SAFE patches to improve a local RAG+autonomous-agent codebase.\n"
    "Constraints:\n"
    "1) Only modify files explicitly allowed.\n"
    "2) Keep patches minimal and valid Python; avoid heavy deps.\n"
    "3) Prefer improvements that: (a) increase robustness; (b) improve crawling/ingestion quality; "
    "(c) strengthen retrieval and memory use; (d) reduce crashes on Windows/Python 3.14; "
    "(e) keep SAFE_MODE semantics consistent; (f) improve logging and error messages where helpful.\n"
    "4) Output a single unified diff patch rooted at the project root. No commentary—only the diff.\n"
    "5) If you include prose accidentally, still include a valid unified diff block somewhere."
)

USER_TEMPLATE = (
    "Project context (truncated):\n{context}\n\n"
    "Goal: {goal}\n\n"
    "Output ONLY a unified diff patch that edits allowed files: "
    + ", ".join(sorted(ALLOWED_EDIT)) +
    ". Keep total output under {max_chars} characters."
)

def _ollama_chat(messages: List[dict], timeout_s: int = 180, max_retries: int = 3) -> str:
    delay = 2.0
    last_err = None
    for _ in range(max_retries):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "messages": messages, "stream": False},
                timeout=timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2.0, 15.0)
    raise RuntimeError(f"Ollama request failed: {last_err}")

def ask_ollama(goal: str, context: str) -> str:
    prompt = USER_TEMPLATE.format(context=context, goal=goal, max_chars=MAX_PATCH_CHARS)
    return _ollama_chat(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",   "content": prompt}],
        timeout_s=180,
        max_retries=3,
    )

# -------------------- Diff extraction / validation ---------------------
_DIFF_HEADER_RE = re.compile(r"^(?:diff --git .+|---\s+[ab/].+|\+\+\+\s+[ab/].+|Index:\s+.+)$", re.M)

def _extract_first_unified_diff(text: str) -> Optional[str]:
    """Pull first usable unified diff from arbitrary model output."""
    if not text:
        return None
    # prefer fenced block if present
    fence = re.search(r"```(?:diff)?\s*\n(.*?)```", text, flags=re.S | re.I)
    candidate = fence.group(1) if fence else text
    m = _DIFF_HEADER_RE.search(candidate)
    if not m:
        return None
    start = m.start()
    chunk = candidate[start:]
    if len(chunk) > MAX_PATCH_CHARS:
        chunk = chunk[:MAX_PATCH_CHARS]
    # must contain both +++ and ---
    if ("+++" not in chunk) or ("---" not in chunk):
        return None
    return chunk.strip()

def _write_attempt(text: str, kind: str) -> pathlib.Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pf = PROPOSALS / f"proposal_{ts}{('.patch' if kind == 'patch' else '.raw.txt')}"
    pf.write_text(text, encoding="utf-8", newline="\n")
    return pf

def _paths_from_diff_headers(patch_text: str) -> List[str]:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith(("+++ ", "--- ")):
            p = line.split("\t", 1)[0].replace("+++ ", "").replace("--- ", "")
            if p.startswith(("a/", "b/")):
                p = p[2:]
            paths.append(p.strip())
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                p = parts[3].strip()
                if p.startswith(("a/", "b/")):
                    p = p[2:]
                paths.append(p)
        elif line.startswith("Index: "):
            p = line.replace("Index: ", "").strip()
            paths.append(p)
    return paths

def _is_safe_basename(name: str) -> bool:
    # only base names allowed (no subpaths)
    bn = os.path.basename(name)
    return bn == name

def patch_touches_only_allowed(patch_text: str) -> bool:
    allowed = set(ALLOWED_EDIT)
    blocked = set(BLOCKLIST)
    paths = _paths_from_diff_headers(patch_text)
    if not paths:
        return False
    for p in paths:
        if not _is_safe_basename(p):
            return False
        bn = os.path.basename(p)
        if bn in blocked:
            return False
        if bn != "/dev/null" and bn not in allowed:
            return False
    return True

# ----------------- Apply / rollback / tests / static -------------------
def _git_apply_check(patch_path: pathlib.Path) -> bool:
    chk = _run(["git", "apply", "--check", str(patch_path)])
    return chk.returncode == 0

def apply_patch(patch_path: pathlib.Path) -> bool:
    if git_available():
        if not _git_apply_check(patch_path):
            return False
        r = _run(["git", "apply", "--whitespace=fix", str(patch_path)])
        if r.returncode != 0:
            return False
        _run(["git", "add", "."])
        _run(["git", "commit", "-m", f"auto-improve: apply {patch_path.name}"])
        return True
    # Fallback: try 'patch' if present (often not on Windows)
    try:
        r = _run(["patch", "-p0", "-i", str(patch_path)])
        return r.returncode == 0
    except Exception:
        return False

def run_static_check() -> bool:
    if not STATIC_CHECK:
        return True
    r = _run(STATIC_CHECK.split())
    return r.returncode == 0

def run_smoke_tests() -> bool:
    if not RUN_TESTS:
        return True
    r = _run(["py", "tests/smoke.py"])
    return r.returncode == 0

def revert_last_commit():
    if git_available():
        _run(["git", "reset", "--hard", "HEAD~1"])

# ------------------------------ Cycle ----------------------------------
def cycle(goal: str) -> str:
    # Optional gate: patch only when new agent memories appear
    if ONLY_ON_NEW_MEMORIES:
        latest_ts = _latest_memory_ts()
        last_applied_ts = _load_last_applied_ts()
        if latest_ts <= last_applied_ts:
            return "NO_NEW_INFO"

    context = collect_context()
    try:
        raw = ask_ollama(goal, context)
    except Exception as e:
        (LOGS / "ollama_error.log").write_text(str(e), encoding="utf-8")
        return "OLLAMA_UNAVAILABLE"

    if not raw or not raw.strip():
        return "NO_PATCH"

    diff = _extract_first_unified_diff(raw)
    if not diff:
        _write_attempt(raw, "raw")
        return "NO_PATCH"

    if len(diff) > MAX_PATCH_CHARS:
        return "PATCH_TOO_BIG"
    if not patch_touches_only_allowed(diff):
        _write_attempt(diff, "raw")
        return "PATCH_TOUCHED_FORBIDDEN"

    pf = _write_attempt(diff, "patch")

    backup_dir = None
    if not git_available():
        backup_dir = snapshot_backup()

    if not apply_patch(pf):
        if backup_dir:
            restore_snapshot(backup_dir)
        return "APPLY_FAILED"

    if not run_static_check() or not run_smoke_tests():
        if git_available():
            revert_last_commit()
        elif backup_dir:
            restore_snapshot(backup_dir)
        return "TESTS_FAILED"

    # success
    summary = {
        "time": _now_iso(),
        "patch": pf.name,
        "result": "APPLIED",
        "goal": goal,
    }
    (PROPOSALS / (pf.stem + ".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    latest_ts = _latest_memory_ts()
    if latest_ts:
        _store_last_applied_ts(latest_ts)

    return "OK"

# -------------------------------- Main --------------------------------
def _parse_cli():
    # Minimal flag parser (no argparse to keep dependencies zero)
    flags = {"once": False, "goal": None, "interval": None, "only_on_new": None}
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--once":
            flags["once"] = True
        elif a == "--goal" and i + 1 < len(args):
            flags["goal"] = args[i + 1]; i += 1
        elif a == "--interval" and i + 1 < len(args):
            try: flags["interval"] = max(30, int(args[i + 1])); i += 1
            except: pass
        elif a == "--only-on-new":
            flags["only_on_new"] = True
        i += 1
    return flags

def main():
    flags = _parse_cli()
    git_init_if_needed()

    # effective settings
    goal = flags["goal"] or os.environ.get("CLEU_AUTO_GOAL") or (
        "Improve self-learning and code efficiency using computer-science patterns discovered through ingestion. "
        "Refactor for stability/speed on Windows + Python 3.14, strengthen retrieval/memory integration, "
        "and enhance error messages while keeping patches small and safe."
    )
    interval = flags["interval"] or int(os.environ.get("CLEU_AUTO_INTERVAL", CYCLE_SECONDS))
    interval = max(30, interval)  # floor
    use_only_on_new = (flags["only_on_new"] is True) or ONLY_ON_NEW_MEMORIES

    _status("Auto-Improver running. Ctrl+C to stop.")
    _status(f"Model={MODEL_NAME}  Interval={interval}s  OnlyOnNew={use_only_on_new}  Git={git_available()}")

    try:
        if flags["once"]:
            status = cycle(goal)
            _status(f"[auto] {datetime.now(UTC).strftime('%H:%M:%S')} -> {status}")
            return
        while True:
            status = cycle(goal)
            _status(f"[auto] {datetime.now(UTC).strftime('%H:%M:%S')} -> {status}")
            time.sleep(interval)
    except KeyboardInterrupt:
        _status("Received Ctrl+C — stopping cleanly.")
    except Exception as e:
        _status(f"Fatal error: {e}")

if __name__ == "__main__":
    main()

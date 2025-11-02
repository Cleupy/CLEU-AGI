# agent_multi.py — Three-agent loop (Coder → Critic → Planner) for patch proposals
# - Windows/Python 3.14 friendly
# - Uses your local Ollama models
# - Writes artifacts into ./proposals and ./auto_logs
# - Never auto-applies patches (safe-by-default). You can apply via git or auto_improve.py.

import os, json, time, pathlib, argparse, subprocess, textwrap
from datetime import datetime, UTC
from typing import List, Dict, Optional
import requests

# ==== Config (inherits your conventions) =====================================

# You can override any of these with environment variables.
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

MODEL_CODER    = os.environ.get("CLEU_MODEL_CODER",   "qwen2.5-coder:7b")
MODEL_CRITIC   = os.environ.get("CLEU_MODEL_CRITIC",  "mistral:7b-instruct")
MODEL_PLANNER  = os.environ.get("CLEU_MODEL_PLANNER", "mistral:7b-instruct")

# Only basenames are honored (to match your auto_improver guardrail style)
ALLOWED_EDIT   = set(os.environ.get("CLEU_ALLOWED_EDIT",
                     "agent.py,agi_core.py,ingest.py,memory_store.py,main.py").split(","))

MAX_PATCH_CHARS = int(os.environ.get("CLEU_MAX_PATCH_CHARS", "40000"))

ROOT       = pathlib.Path(__file__).resolve().parent
PROPOSALS  = ROOT / "proposals"
LOGS       = ROOT / "auto_logs"
STORE      = ROOT / "store"
PROPOSALS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def _log_path(name: str) -> pathlib.Path:
    return LOGS / name

def _append_log(name: str, line: str):
    with open(_log_path(name), "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

# ==== Small utilities =========================================================

def read_file_head(basename: str, limit: int = 16000) -> str:
    p = ROOT / basename
    if not p.exists() or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")[:limit]
    except Exception:
        return ""

def collect_code_context() -> str:
    parts = []
    for bn in sorted(ALLOWED_EDIT):
        bn = bn.strip()
        if not bn: 
            continue
        head = read_file_head(bn)
        if head:
            parts.append(f"== {bn} ==\n{head}")
    return "\n\n".join(parts) if parts else "(no allowed files found)"

def git_available() -> bool:
    try:
        r = subprocess.run(["git","--version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, shell=False, cwd=str(ROOT))

# ==== Ollama chat wrapper with retries ========================================

def ollama_chat(model: str, messages: List[Dict[str,str]], timeout_s: int = 180, retries: int = 3) -> str:
    last = None
    for _ in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={"model": model, "messages": messages, "stream": False}, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            last = e
            time.sleep(2.5)
    raise RuntimeError(f"Ollama failed after retries: {last}")

# ==== Prompts =================================================================

SYSTEM_CODER = (
    "You are the Coder. Produce a SMALL, SAFE unified diff patch against a local codebase.\n"
    "Constraints:\n"
    "1) Edit only these basenames: " + ", ".join(sorted(ALLOWED_EDIT)) + "\n"
    "2) Keep patch valid unified diff (diff --git / --- / +++ / @@ hunks).\n"
    "3) No heavy deps; avoid large refactors. Max size: {max_chars} chars.\n"
    "4) Return ONLY the diff. No explanations."
)

SYSTEM_CRITIC = (
    "You are the Critic. Review a unified diff for safety and quality.\n"
    "Return JSON with keys: ok (bool), reasons (list of strings), suggestions (list of strings), "
    "and allowed (bool indicating only allowed basenames were touched)."
)

SYSTEM_PLANNER = (
    "You are the Planner. Given a task, the code context, and reviewer feedback, produce:\n"
    "1) 'next_goals': 3 concrete micro-goals to improve the system soon\n"
    "2) 'commit_message': a single-line git commit msg if the patch is acceptable (else '')\n"
    "3) 'accept': true/false whether to accept the patch as-is\n"
    "Return strict JSON with keys: next_goals, commit_message, accept."
)

USER_CODER_TEMPLATE = (
    "Task:\n{task}\n\n"
    "Code context (truncated):\n{context}\n\n"
    "Generate a patch now. Remember: ONLY allowed files. Only the diff."
)

USER_CRITIC_TEMPLATE = (
    "Here is the diff to review:\n\n```\n{diff}\n```\n\n"
    "Check that:\n- it is valid unified diff\n- modifies only allowed files: " + ", ".join(sorted(ALLOWED_EDIT)) + "\n"
    "- changes are minimal and plausible\n- no secrets or destructive ops\n- within {max_chars} chars"
)

USER_PLANNER_TEMPLATE = (
    "Task:\n{task}\n\n"
    "Reviewer feedback:\n{review_json}\n\n"
    "Code context (truncated):\n{context}\n\n"
    "Respond with JSON only."
)

# ==== Diff validators ==========================================================

import re
DIFF_HEADER_RE = re.compile(r"^(?:diff --git .+|---\s+[ab/].+|\+\+\+\s+[ab/].+|Index:\s+.+)$", re.M)

def extract_first_unified_diff(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer fenced block if present
    m = re.search(r"```(?:diff)?\s*\n(.*?)```", text, flags=re.S|re.I)
    cand = m.group(1) if m else text
    m2 = DIFF_HEADER_RE.search(cand)
    if not m2:
        return None
    chunk = cand[m2.start():]
    if len(chunk) > MAX_PATCH_CHARS:
        chunk = chunk[:MAX_PATCH_CHARS]
    if ("+++" not in chunk) or ("---" not in chunk):
        return None
    return chunk.strip()

def diff_only_touches_allowed(diff_text: str) -> bool:
    paths = []
    for line in diff_text.splitlines():
        if line.startswith(("+++ ", "--- ")):
            p = line.split("\t", 1)[0].replace("+++ ", "").replace("--- ", "")
            if p.startswith(("a/","b/")): p = p[2:]
            paths.append(p.strip())
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                p = parts[3]
                if p.startswith(("a/","b/")): p = p[2:]
                paths.append(p.strip())
        elif line.startswith("Index: "):
            paths.append(line.replace("Index: ", "").strip())
    if not paths:
        return False
    for p in paths:
        bn = os.path.basename(p)
        if bn not in ALLOWED_EDIT and bn != "/dev/null":
            return False
    return True

# ==== Save artifacts ===========================================================

def save_text(folder: pathlib.Path, stem: str, ext: str, content: str) -> pathlib.Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{stem}_{ts}{ext}"
    path.write_text(content, encoding="utf-8")
    return path

# ==== Main orchestration =======================================================

def multi_agent_once(task: str, dry_run_apply: bool = True) -> Dict:
    """Run Coder → Critic → Planner once. Returns a result dict and writes artifacts."""
    context = collect_code_context()

    # 1) Coder
    coder_out = ollama_chat(
        MODEL_CODER,
        messages=[
            {"role":"system","content": SYSTEM_CODER.format(max_chars=MAX_PATCH_CHARS)},
            {"role":"user",  "content": USER_CODER_TEMPLATE.format(task=task, context=context)}
        ],
        timeout_s=240, retries=3
    )
    diff = extract_first_unified_diff(coder_out)
    if not diff:
        save_text(PROPOSALS, "multi_raw_no_diff", ".txt", coder_out)
        return {"ok": False, "stage": "coder", "reason": "NO_DIFF_EXTRACTED"}

    if len(diff) > MAX_PATCH_CHARS:
        save_text(PROPOSALS, "multi_raw_too_big", ".patch", diff)
        return {"ok": False, "stage": "coder", "reason": "PATCH_TOO_BIG"}

    if not diff_only_touches_allowed(diff):
        save_text(PROPOSALS, "multi_raw_forbidden", ".patch", diff)
        return {"ok": False, "stage": "coder", "reason": "PATCH_TOUCHES_FORBIDDEN"}

    patch_path = save_text(PROPOSALS, "multi_proposed", ".patch", diff)

    # 2) Critic
    review_text = ollama_chat(
        MODEL_CRITIC,
        messages=[
            {"role":"system","content": SYSTEM_CRITIC},
            {"role":"user",  "content": USER_CRITIC_TEMPLATE.format(diff=diff, max_chars=MAX_PATCH_CHARS)}
        ],
        timeout_s=120, retries=3
    )
    # be permissive—best effort to parse JSON
    try:
        review = json.loads(review_text)
    except Exception:
        review = {"ok": False, "reasons": ["Reviewer did not return valid JSON"], "suggestions": [], "allowed": False}
    save_text(PROPOSALS, "multi_review", ".json", json.dumps(review, ensure_ascii=False, indent=2))

    # 3) Planner
    plan_text = ollama_chat(
        MODEL_PLANNER,
        messages=[
            {"role":"system","content": SYSTEM_PLANNER},
            {"role":"user",  "content": USER_PLANNER_TEMPLATE.format(task=task, review_json=json.dumps(review, ensure_ascii=False, indent=2), context=context)}
        ],
        timeout_s=120, retries=3
    )
    try:
        plan_json = json.loads(plan_text)
    except Exception:
        plan_json = {"next_goals": [], "commit_message": "", "accept": False}
    save_text(PROPOSALS, "multi_plan", ".json", json.dumps(plan_json, ensure_ascii=False, indent=2))

    # 4) Optional: git dry-run apply
    apply_ok, apply_err = True, ""
    if dry_run_apply and git_available():
        chk = run(["git","apply","--check", str(patch_path)])
        if chk.returncode != 0:
            apply_ok = False
            apply_err = chk.stderr
            save_text(PROPOSALS, "multi_git_apply_check_error", ".log", apply_err)

    result = {
        "ok": True,
        "stage": "done",
        "patch": str(patch_path),
        "review_ok": bool(review.get("ok")),
        "review_allowed": bool(review.get("allowed")),
        "plan_accept": bool(plan_json.get("accept", False)),
        "plan_next_goals": plan_json.get("next_goals", []),
        "plan_commit_message": plan_json.get("commit_message", ""),
        "git_dry_run_ok": apply_ok,
        "git_dry_run_error": apply_err,
    }
    return result

# ==== CLI ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Run the 3-agent patch proposer once or in a loop.")
    ap.add_argument("--task", required=True, help="Natural-language task (e.g., 'Add /goodbye help alias in main.py').")
    ap.add_argument("--every", type=int, default=0, help="Minutes between runs (0 = run once).")
    ap.add_argument("--no-dry-run", action="store_true", help="Skip git --check dry-run.")
    args = ap.parse_args()

    interval = max(0, args.every)
    dry = not args.no_dry_run

    print(f"Three-agent loop starting with:\n  Coder={MODEL_CODER}\n  Critic={MODEL_CRITIC}\n  Planner={MODEL_PLANNER}\n  Interval(min)={interval}\n  DryRunGit={dry}\n")

    def one():
        res = multi_agent_once(args.task, dry_run_apply=dry)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        _append_log("agent_multi.log", _now_iso() + " " + json.dumps(res, ensure_ascii=False))

        # If everything looks green and planner accepts, also save a ready-to-commit note
        if res.get("ok") and res.get("plan_accept") and res.get("git_dry_run_ok"):
            msg = res.get("plan_commit_message") or f"auto: implement task — {args.task}"
            save_text(PROPOSALS, "multi_commit_message", ".txt", msg)

    if interval == 0:
        one()
        return
    try:
        while True:
            one()
            for _ in range(interval * 60):
                time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()

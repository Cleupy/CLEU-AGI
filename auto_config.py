# auto_config.py — knobs for the self-coding loop

# ---- Model / endpoint ----
# Code-tuned models emit diffs more reliably than general chat models.
MODEL_NAME  = "qwen2.5-coder:7b"                 # must appear in `ollama list`
OLLAMA_URL  = "http://localhost:11434/api/chat"  # default Ollama local API

# ---- Schedule ----
CYCLE_SECONDS = 600  # how often to attempt a self-improvement cycle (seconds)

# ---- What files the improver may edit (basenames only) ----
ALLOWED_EDIT = {
    "agent.py",
    "agi_core.py",
    "ingest.py",
    "memory_store.py",
    "main.py",
}

# ---- Files it must never touch (basenames only) ----
# The improver checks basenames of diff paths; blocking directories here
# only helps if a diff header literally uses that basename.
BLOCKLIST = {
    "auto_improve.py",
    "auto_config.py",
    "tests/smoke.py",
    "requirements.txt",
    "README.md",
    # persistent data / logs / proposals (protect by basename)
    "embeddings.npy",
    "meta.jsonl",
    "memory_log.jsonl",
    "proposals",
    "auto_logs",
    "store",
    "logs",
}

# ---- Patch size cap (characters). Prevents wild refactors. ----
MAX_PATCH_CHARS = 40000

# ---- Safety checks before accepting a patch ----
# Set RUN_TESTS=False unless you actually have tests/smoke.py
RUN_TESTS    = False                 # set True only if tests/smoke.py exists
USE_GIT      = True                  # use git for apply/rollback when available
# Keep STATIC_CHECK off unless pyflakes is installed: `py -m pip install pyflakes`
STATIC_CHECK = ""                    # e.g., "py -m pyflakes" to enable

# ---- OPTIONAL: Only patch when new info arrived from main.py ----
# If True, the improver will skip cycles until `store/memory_log.jsonl`
# has newer entries than at the time of the last successful patch.
ONLY_ON_NEW_MEMORIES = False

# How many recent lines to scan in memory_log/meta to detect new info.
MEMORY_DELTA_WINDOW = 400

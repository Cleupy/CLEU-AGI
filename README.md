# CLEU‑AGI Lab (minimal, local, modular)

A tiny, modular playground to build a ChatGPT‑like assistant with:
- Local **retrieval** (vector search using scikit‑learn HashingVectorizer).
- **Web ingestion** with SAFE_MODE allowlist.
- Optional **Ollama** LLM (fallback to retrieval‑only answer if not available).
- **Reflection** and **goal** scaffold to experiment with AGI‑style loops.

> Runs on Windows with Python 3.11+ (tested on 3.14). No PyTorch required.

## 1) Install deps
```powershell
cd "<your folder>"
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## 2) (Optional) Start Ollama and pull a model
```powershell
# In a separate terminal (only if you want LLM answers)
ollama serve
ollama pull mistral:7b-instruct
```

## 3) Run
```powershell
py main.py
```

## 4) Commands (inside the app)
```
/help                     # show commands
/exit                     # quit
/ingest <url> [--max N]   # fetch & index a page (and up to N internal links)
/ask <question>           # answer using retrieval + (optional) Ollama
/recall <query>           # show top matching chunks
/plan <goal>              # make a tiny plan (stores to memory)
/reflect <text>           # run a self-check on a statement/answer
/safe on|off              # toggle SAFE_MODE (off = your risk)
```

## 5) Storage
- `store/embeddings.npy`   — dense float32 vectors (HashingVectorizer -> dense).
- `store/meta.jsonl`       — one JSON per ingested chunk.
- `store/memory.jsonl`     — your long‑term facts/plans/reflections.

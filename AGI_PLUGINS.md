# CLEU-AGI Plugins (Models + Agents)

## Models (Ollama)
- **Reasoning/General**: `llama3.1:8b` (open-weights, Meta community license).
- **Code editing**: `qwen2.5-coder:7b` (strong diff emission).
- Optional: `mistral:7b-instruct` as a lightweight backup.

> Switch by editing `auto_config.py::MODEL_NAME` and/or setting `OLLAMA_MODEL` env var.

## Agent Orchestration
- **AutoGen (Microsoft)** — Multi-agent orchestration, tool calling, human-in-the-loop.
  - Use CLEU tools: `search`, `ingest`, `apply_patch`, `run_tests`, `revert`.
  - Pattern: Critic (review patch) → Coder (emit diff) → Executor (apply + test).

## Safety & License Notes
- **Llama** is open-weights under a community license (not OSI). Review terms before commercial use.
- Keep network access gated by allow-listed tools; log URLs + SHA256 of retrieved content.

## Hardening To-Dos
- Normalize line endings to LF repo-wide (`git config core.eol lf`).
- Use `git apply --check` before every apply (already in `auto_improve.py`).
- Keep `RUN_TESTS=True` and add more unit tests for critical paths.

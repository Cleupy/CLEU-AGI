# agi_core.py — Model wrapper (Ollama optional)
import os, requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")

SYSTEM_PROMPT = (
    "You are CLEU‑AGI: a careful, factual assistant. Prefer provided context. "
    "If unknown, say so briefly."
)

def ask_llm(user_prompt: str, context_text: str = "") -> str:
    """Try Ollama. If not available, raise to let caller fallback."""
    prompt = (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User:\n{user_prompt}\n\n"
        f"Assistant:\n"
    )
    payload = {"model": MODEL_NAME, "messages": [{"role":"user","content": prompt}], "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()
    return "[no response]"

# memory_store.py — embeddings + memory (no Torch)
import os, json, time, numpy as np
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import HashingVectorizer

STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)

EMB_PATH  = os.path.join(STORE_DIR, "embeddings.npy")   # (N, D) float32
META_PATH = os.path.join(STORE_DIR, "meta.jsonl")       # chunks metadata
MEM_PATH  = os.path.join(STORE_DIR, "memory.jsonl")     # user facts/plans

# Embedding backend
_hv = HashingVectorizer(n_features=4096, ngram_range=(1,2), alternate_sign=False, norm=None)
_DIM = 4096

def _encode(texts: List[str]):
    X = _hv.transform(texts)
    dense = X.astype(np.float32).toarray()
    n = np.linalg.norm(dense, axis=1, keepdims=True)
    n[n==0] = 1.0
    return dense / n

def _load_embeddings() -> np.ndarray:
    if os.path.exists(EMB_PATH):
        arr = np.load(EMB_PATH)
        return arr.astype(np.float32, copy=False).reshape(-1, _DIM)
    return np.zeros((0, _DIM), dtype=np.float32)

def _save_embeddings(arr: np.ndarray):
    np.save(EMB_PATH, arr.astype(np.float32, copy=False))

def _append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out

def add_chunks(chunks: List[str], meta_proto: Dict[str, Any]) -> int:
    if not chunks: return 0
    vecs = _encode(chunks)
    E = _load_embeddings()
    E = np.vstack([E, vecs]) if E.size else vecs
    _save_embeddings(E)
    ts = time.time()
    for c in chunks:
        m = dict(meta_proto)
        m.update({"snippet": c[:500], "ts": ts})
        _append_jsonl(META_PATH, m)
    return len(chunks)

def search(query: str, k: int = 6) -> List[Dict[str, Any]]:
    metas = _load_jsonl(META_PATH)
    E = _load_embeddings()
    if E.shape[0] == 0 or not metas: return []
    qv = _encode([query])[0:1, :]
    sims = (E @ qv.T).ravel()
    k = min(k, sims.size)
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    out = []
    for i in idx:
        if 0 <= i < len(metas):
            m = dict(metas[i])
            m["score"] = float(sims[i])
            out.append(m)
    return out

def remember(kind: str, text: str):
    _append_jsonl(MEM_PATH, {"kind": kind, "text": text, "ts": time.time()})

def recall_mem(kind: str = None) -> List[Dict[str, Any]]:
    rows = _load_jsonl(MEM_PATH)
    if kind:
        rows = [r for r in rows if r.get("kind")==kind]
    return rows[-50:]  # recent

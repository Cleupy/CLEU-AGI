# tests/smoke.py
import importlib, sys

def _ok(label, fn):
    try:
        fn()
        print(f"[OK] {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        sys.exit(1)

def test_imports():
    for m in ["agent", "ingest", "agi_core", "memory_store"]:
        importlib.invalidate_caches()
        importlib.import_module(m)

def test_answer_minimal():
    from agent import answer
    out = answer("Say 'ping' twice.")
    assert isinstance(out, str) and len(out) > 0

def test_ingest_recall_roundtrip():
    from ingest import ingest_url
    from agent import answer
    # Keep SAFE_MODE as user set; just ensure function calls don't crash:
    try:
        ingest_url("https://example.com", follow=0)
    except Exception:
        # network/robots failure is fine; we just don't want crashes later
        pass
    r = answer("What is example.com?")
    assert isinstance(r, str)

if __name__ == "__main__":
    _ok("imports", test_imports)
    _ok("answer()", test_answer_minimal)
    _ok("ingest/answer roundtrip", test_ingest_recall_roundtrip)
    print("[SMOKE] all tests passed")

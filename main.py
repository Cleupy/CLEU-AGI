# main.py — CLEU-AGI interactive console + autonomous mode
import os, shlex, sys, traceback, time
from agent import answer, plan, reflect, history, autonomous_cycle
from ingest import ingest_url, SAFE_MODE

BANNER = """
╔══════════════════════════════════════════════╗
║          CLEU-AGI: Self-Evolving Agent       ║
║         Mission: Learn. Reason. Reflect.     ║
╚══════════════════════════════════════════════╝
Type /help for commands. /exit to quit.
"""

def show_help():
    print("""
Commands
  /help                         Show this help
  /exit                         Quit
  /ingest <url> [--follow N]    Fetch page and index (+ follow N internal links)
  /ask <question>               Ask a factual or reasoning question
  /recall <query>               Show best matching memory chunks
  /plan <goal>                  Create a short goal plan
  /reflect <text>               Self-evaluate a statement or idea
  /mem [kind]                   Show stored memories (answer/plan/reflection)
  /safe on|off                  Toggle SAFE_MODE (limits to trusted domains)
  /auto [minutes]               Enable periodic autonomous learning/reflection
  /stopauto                     Stop autonomous loop if running
""")

# Track autonomous loop thread
_auto_running = False

def run_autonomous_loop(interval: int = 30):
    """Run CLEU-AGI in a repeating autonomous mode."""
    global _auto_running
    _auto_running = True
    print(f"[AUTO] CLEU-AGI autonomous mode engaged — every {interval} minutes.")
    try:
        while _auto_running:
            try:
                autonomous_cycle()
            except Exception as e:
                print(f"[AUTO ERROR] {e}")
                traceback.print_exc()
            print(f"[AUTO] Sleeping {interval} minutes…\n")
            for _ in range(interval * 60):
                if not _auto_running:
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        print("[AUTO] Interrupted by user.")
    finally:
        _auto_running = False
        print("[AUTO] Autonomous mode stopped.")

def stop_autonomous_loop():
    global _auto_running
    if _auto_running:
        _auto_running = False
        print("[AUTO] Stopping autonomous loop…")
    else:
        print("[AUTO] No autonomous loop active.")

def main():
    global SAFE_MODE
    print(BANNER)

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[System] Shutdown signal received. Goodbye.")
            break

        if not line:
            continue

        # Normal conversation (no slash)
        if not line.startswith("/"):
            try:
                print(answer(line))
            except Exception as e:
                print(f"[Error] {e}")
            continue

        # Command mode
        try:
            parts = shlex.split(line)
        except Exception:
            print("[Parse Error] Invalid command format.")
            continue

        cmd, args = parts[0].lower(), parts[1:]

        # === COMMAND HANDLERS ===
        if cmd == "/exit":
            stop_autonomous_loop()
            break

        elif cmd == "/help":
            show_help()

        elif cmd == "/ask":
            print(answer(" ".join(args)))

        elif cmd == "/plan":
            print(plan(" ".join(args)))

        elif cmd == "/reflect":
            print(reflect(" ".join(args)))

        elif cmd == "/mem":
            print(history(args[0] if args else None))

        elif cmd == "/safe":
            if not args:
                print(f"SAFE_MODE is {'on' if SAFE_MODE else 'off'}")
            else:
                val = args[0].lower() in ("on", "true", "1", "yes")
                import ingest
                ingest.SAFE_MODE = val
                SAFE_MODE = val
                print(f"SAFE_MODE → {'on' if val else 'off'}")

        elif cmd == "/ingest":
            if not args:
                print("Usage: /ingest <url> [--follow N]")
                continue
            url, follow = None, 0
            i = 0
            while i < len(args):
                if args[i] == "--follow" and i + 1 < len(args):
                    try:
                        follow = int(args[i + 1]); i += 2; continue
                    except Exception:
                        pass
                if url is None:
                    url = args[i]
                i += 1
            try:
                n = ingest_url(url, follow)
                print(f"[Ingested] {n} chunks/links.")
            except Exception as e:
                print(f"[Ingest Failed] {e}")

        elif cmd == "/auto":
            interval = int(args[0]) if args else 30
            run_autonomous_loop(interval)

        elif cmd == "/stopauto":
            stop_autonomous_loop()

        else:
            print("[Unknown] Type /help for available commands.")

if __name__ == "__main__":
    main()

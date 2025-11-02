# memory_graph.py — tiny triple store over networkx
import os, time, json
from typing import Iterable, List, Dict, Optional
import networkx as nx

STORE_DIR = os.path.join(os.getcwd(), "store")
os.makedirs(STORE_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(STORE_DIR, "graph.gpickle")

def _load() -> nx.MultiDiGraph:
    if os.path.exists(GRAPH_PATH):
        try:
            return nx.read_gpickle(GRAPH_PATH)
        except Exception:
            pass
    return nx.MultiDiGraph()  # nodes: entities; edges: (s)-[p]->(o)

def _save(g: nx.MultiDiGraph):
    nx.write_gpickle(g, GRAPH_PATH)

def add_triple(s: str, p: str, o: str, source: str = "manual"):
    g = _load()
    g.add_node(s)
    g.add_node(o)
    g.add_edge(s, o, key=f"{p}-{time.time():.6f}", predicate=p, ts=time.time(), source=source)
    _save(g)

def triples_about(entity: str, direction: str = "out") -> List[Dict]:
    """
    direction: 'out' -> (entity -> *), 'in' -> (* -> entity), 'both'
    """
    g = _load()
    rows = []
    if direction in ("out", "both"):
        for _, o, key, data in g.out_edges(entity, keys=True, data=True):
            rows.append({"s": entity, "p": data.get("predicate",""), "o": o, "source": data.get("source","")})
    if direction in ("in", "both"):
        for s, _, key, data in g.in_edges(entity, keys=True, data=True):
            rows.append({"s": s, "p": data.get("predicate",""), "o": entity, "source": data.get("source","")})
    return rows

def find_path(a: str, b: str, max_len: int = 3) -> List[str]:
    """Return one simple path a→…→b up to max_len using unlabelled edges."""
    g = _load()
    try:
        for length in range(1, max_len + 1):
            for path in nx.all_simple_paths(g.to_undirected(), source=a, target=b, cutoff=length):
                return path
    except Exception:
        pass
    return []

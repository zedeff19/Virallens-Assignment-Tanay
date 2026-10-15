"""Q2 – domain-specific embedder for the securities-law corpus.

Run from the repo root:  python -m q2_embedder_final.main
The repo root is put on sys.path so the Q1 ingestion code (`src.ingest`, `src.retrieval`) is reused.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

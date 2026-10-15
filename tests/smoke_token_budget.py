"""Live smoke test for the Groq token budget (not collected by pytest). Costs ~15 real Groq calls.

Run manually:  python tests/smoke_token_budget.py [N]        (N calls per arm, default 10)

Fires N rapid calls carrying DeepAgents' default filesystem tool schemas (~2.6K tokens each) in two
arms, counting real HTTP 429s by wrapping the Groq SDK's `Completions.create`:

  A. before  plain ChatGroq + the old requests-only InMemoryRateLimiter, direct calls
  B. after   get_llm_with_fallbacks("retriever"): token bucket + retry + fallback (Groq models only)

Only GROQ_API_KEY is used (the other providers are removed from the environment so their exhausted
free tiers are neither touched nor counted). The arms are separated by 70s so the TPM bucket refills.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import groq  # noqa: E402
from groq.resources.chat.completions import Completions  # noqa: E402

from src import llm  # noqa: E402  (load_dotenv runs on import)

for var in ("GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
    os.environ.pop(var, None)
if not os.getenv("GROQ_API_KEY"):
    sys.exit("GROQ_API_KEY not set")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ROLE = "retriever"
MODEL = llm.get_chain(ROLE)[0].model

STATS: dict[str, Counter] = {}
ARM = "setup"
_orig_create = Completions.create


def _counting_create(self, *args, **kwargs):
    c = STATS.setdefault(ARM, Counter())
    c[f"http_calls[{kwargs.get('model')}]"] += 1
    try:
        return _orig_create(self, *args, **kwargs)
    except groq.RateLimitError as exc:
        c["http_429"] += 1
        c["last_429"] = str(exc)[:160]
        raise


Completions.create = _counting_create


def default_tools() -> list:
    from deepagents.middleware.filesystem import FilesystemMiddleware

    return list(FilesystemMiddleware().tools)


def run_arm(name: str, runnable) -> list[dict]:
    global ARM
    ARM = name
    rows = []
    t_arm = time.monotonic()
    for i in range(N):
        t0 = time.monotonic()
        row: dict = {"i": i + 1}
        try:
            out = runnable.invoke(f"[{name} {i}] Reply with the single word: ok")
            meta = getattr(out, "response_metadata", {}) or {}
            usage = getattr(out, "usage_metadata", None) or {}
            row.update(ok=True, model=meta.get("model_name", "?"), tokens=usage.get("total_tokens"))
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:110]}")
        row["secs"] = time.monotonic() - t0
        rows.append(row)
        status = f"ok via {row['model']} ({row['tokens']} tok)" if row["ok"] else f"FAIL {row['error']}"
        print(f"  [{name}] {i + 1:>2}/{N}  {row['secs']:6.1f}s  {status}", flush=True)
    print(f"  [{name}] total wall time {time.monotonic() - t_arm:.1f}s", flush=True)
    return rows


def summarize(name: str, rows: list[dict]) -> None:
    c = STATS.get(name, Counter())
    ok = [r for r in rows if r["ok"]]
    by_model = Counter(r["model"] for r in ok)
    print(f"\n[{name}] answered {len(ok)}/{len(rows)}  |  HTTP 429s from Groq: {c['http_429']}")
    print(f"  answered by: {dict(by_model)}")
    print(f"  HTTP calls:  { {k: v for k, v in c.items() if k.startswith('http_calls')} }")
    fails = Counter(r["error"].split(':')[0] for r in rows if not r["ok"])
    if fails:
        print(f"  failures:    {dict(fails)}")
    if c["http_429"]:
        print(f"  last 429:    {c['last_429']}")
    waits = sorted(r["secs"] for r in rows)
    print(f"  per-call latency: min {waits[0]:.1f}s  median {waits[len(waits) // 2]:.1f}s  max {waits[-1]:.1f}s")


def main() -> None:
    from langchain_core.messages import HumanMessage
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_groq import ChatGroq

    from src.token_budget import estimate_tokens

    tools = default_tools()
    print(f"model={MODEL}  tools={len(tools)}  calls/arm={N}")

    plain = ChatGroq(
        model=MODEL,
        temperature=0,
        max_retries=0,
        max_tokens=llm._max_completion(MODEL),
        rate_limiter=InMemoryRateLimiter(requests_per_second=llm._RPS["groq"], max_bucket_size=1),
    ).bind_tools(tools)

    # estimator accuracy: one probe call, estimate vs the real prompt tokens Groq reports
    global ARM
    ARM = "probe"
    prompt = "[probe] Reply with the single word: ok"
    out = plain.invoke(prompt)
    actual_in = (out.usage_metadata or {}).get("input_tokens")
    message_dicts, _ = plain.bound._create_message_dicts([HumanMessage(prompt)], None)
    est_prompt = estimate_tokens(message_dicts, plain.kwargs.get("tools"), expected_completion=0)
    print(f"\nestimator check: estimated prompt ~{est_prompt} tok vs Groq-reported input_tokens {actual_in}"
          f"  (ratio est/actual = {est_prompt / actual_in:.2f}; >1 means conservative)\n")

    print("=== Arm A: before (plain ChatGroq, requests-only limiter) ===")
    time.sleep(70)
    rows_a = run_arm("A-before", plain)

    print("\n... waiting 70s for the TPM bucket to refill ...\n")
    time.sleep(70)

    print("=== Arm B: after (get_llm_with_fallbacks: token bucket + retry + fallback) ===")
    chain = llm.get_llm_with_fallbacks(ROLE, lambda m: m.bind_tools(tools))
    rows_b = run_arm("B-after", chain)

    print("\n================ SUMMARY ================")
    summarize("A-before", rows_a)
    summarize("B-after", rows_b)
    bucket = llm._budget(MODEL)
    print(f"\nfinal bucket level for {MODEL}: {bucket.level:.0f} / {bucket.capacity:.0f} tokens")


if __name__ == "__main__":
    main()

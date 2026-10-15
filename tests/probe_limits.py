"""Live probe of each provider's real free-tier limits (not collected by pytest).

Run manually:  python tests/probe_limits.py            (add --billing to also run the max_tokens/TPM
                                                        billing check, which drains one model's bucket)

Makes ONE tiny request per model (about 5 requests in total) with raw HTTP so the response headers are
visible, then prints what each provider reports:

  Groq        x-ratelimit-* headers, per model (limit-requests is RPD, limit-tokens is TPM)
  OpenRouter  GET /api/v1/key (tier, credit limit, usage, rate_limit) plus any rate-limit headers
  Gemini      model token limits from models.get; RPM/RPD are NOT exposed by the API, so this prints the
              quota details of a 429 if one happens and otherwise points you to AI Studio

Keys are read from .env and never printed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.llm import GEMINI_MODEL, GROQ_ALT, GROQ_LARGE, GROQ_SMALL, OPENROUTER_MODEL  # noqa: E402

PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
TIMEOUT = 60
HEADER_HINTS = ("ratelimit", "rate-limit", "retry-after", "x-ratelimit")


def rate_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if any(h in k.lower() for h in HEADER_HINTS)}


def show(title: str, rows: dict[str, object]) -> None:
    print(f"  {title}")
    if not rows:
        print("    (none reported)")
    for k, v in rows.items():
        print(f"    {k:<34} {v}")


def probe_groq() -> None:
    key = os.getenv("GROQ_API_KEY")
    print("\n[groq]")
    if not key:
        print("  SKIPPED (GROQ_API_KEY not set)")
        return
    print("  Note: limit-requests is the DAILY request cap (RPD); limit-tokens is tokens PER MINUTE (TPM).")
    for model in dict.fromkeys([GROQ_LARGE, GROQ_SMALL, GROQ_ALT]):
        print(f"\n  model={model}")
        try:
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": PROMPT, "max_tokens": 16},
                timeout=TIMEOUT,
            )
        except httpx.HTTPError as exc:
            print(f"    FAIL {type(exc).__name__}: {exc}")
            continue
        print(f"    HTTP {r.status_code}")
        show("rate-limit headers", rate_headers(r))
        if r.status_code != 200:
            print(f"    body: {r.text[:300]}")


def _groq_post(key: str, model: str, content: str, max_tokens: int) -> httpx.Response:
    return httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens},
        timeout=TIMEOUT,
    )


def probe_groq_billing() -> None:
    """Does Groq bill the requested max_tokens or the actual usage against TPM? (src/llm.py relies on actual)

    Recorded result (2026-10, free tier, gpt-oss-20b/120b + qwen3.8-27b): a ~78-token prompt with
    max_tokens=9000 (> the 8K TPM) returns 200 and the remaining-tokens header drops only by real usage,
    so max_tokens is NOT billed. This re-checks it, then repeats it against a DRAINED bucket: a few real
    calls with a long prompt first, then a tiny prompt with a large max_tokens must still be accepted.
    Costs ~6 requests on one model (the small one).
    """
    key = os.getenv("GROQ_API_KEY")
    print("\n[groq billing: max_tokens vs actual usage]")
    if not key:
        print("  SKIPPED (GROQ_API_KEY not set)")
        return
    model = GROQ_SMALL
    print(f"  model={model}")
    try:
        r = _groq_post(key, model, "Reply with the single word: ok", 9000)
        verdict = "accepted -> max_tokens not billed" if r.status_code == 200 else f"REJECTED {r.status_code}"
        print(f"  empty bucket, max_tokens=9000: HTTP {r.status_code} ({verdict})")
        if r.status_code != 200:
            print(f"    body: {r.text[:300]}")

        # Drain: ~1.5K-token prompts, 5 calls ~= 7.5K of the 8K TPM budget inside one minute.
        filler = "The quick brown fox jumps over the lazy dog. " * 150
        used = 0
        for i in range(5):
            r = _groq_post(key, model, filler + "\nReply with the single word: ok", 16)
            if r.status_code == 200:
                used += r.json().get("usage", {}).get("total_tokens", 0)
            print(f"  drain {i + 1}/5: HTTP {r.status_code}  tokens used so far ~{used}")
            if r.status_code != 200:
                print(f"    body: {r.text[:300]}")
                break
        show("rate-limit headers after drain", rate_headers(r))

        r = _groq_post(key, model, "Reply with the single word: ok", 9000)
        verdict = (
            "accepted -> max_tokens still not billed on a drained bucket"
            if r.status_code == 200
            else "REJECTED -> max_tokens may count when the bucket is low; revisit src/llm.py"
        )
        print(f"  drained bucket, max_tokens=9000: HTTP {r.status_code} ({verdict})")
        if r.status_code != 200:
            print(f"    body: {r.text[:300]}")
    except httpx.HTTPError as exc:
        print(f"  FAIL {type(exc).__name__}: {exc}")


def probe_openrouter() -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    print("\n[openrouter]")
    if not key:
        print("  SKIPPED (OPENROUTER_API_KEY not set)")
        return
    auth = {"Authorization": f"Bearer {key}"}
    try:
        r = httpx.get("https://openrouter.ai/api/v1/key", headers=auth, timeout=TIMEOUT)
        print(f"  GET /api/v1/key -> HTTP {r.status_code}")
        if r.status_code == 200:
            data = r.json().get("data", {})
            show("key info", {k: data.get(k) for k in ("label", "is_free_tier", "limit", "limit_remaining", "usage", "usage_daily", "rate_limit") if k in data})
        else:
            print(f"    body: {r.text[:300]}")
        r = httpx.get("https://openrouter.ai/api/v1/credits", headers=auth, timeout=TIMEOUT)
        if r.status_code == 200:
            d = r.json().get("data", {})
            bought, used = d.get("total_credits", 0), d.get("total_usage", 0)
            print(f"  credits: bought=${bought} used=${used}  "
                  f"({'>= $10 purchased: 1000 RPD on :free models' if bought >= 10 else '< $10 purchased: 50 RPD on :free models'})")
        print(f"\n  model={OPENROUTER_MODEL}")
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=auth,
            json={"model": OPENROUTER_MODEL, "messages": PROMPT, "max_tokens": 16},
            timeout=TIMEOUT,
        )
        print(f"    HTTP {r.status_code}")
        show("rate-limit headers", rate_headers(r))
        if r.status_code != 200:
            print(f"    body: {r.text[:400]}")
    except httpx.HTTPError as exc:
        print(f"  FAIL {type(exc).__name__}: {exc}")


def probe_gemini() -> None:
    key = os.getenv("GOOGLE_API_KEY")
    print("\n[gemini]")
    if not key:
        print("  SKIPPED (GOOGLE_API_KEY not set)")
        return
    auth = {"x-goog-api-key": key}
    base = "https://generativelanguage.googleapis.com/v1beta"
    print(f"  model={GEMINI_MODEL}")
    try:
        r = httpx.get(f"{base}/models/{GEMINI_MODEL}", headers=auth, timeout=TIMEOUT)
        if r.status_code == 200:
            m = r.json()
            show("model limits", {"inputTokenLimit": m.get("inputTokenLimit"), "outputTokenLimit": m.get("outputTokenLimit")})
        else:
            print(f"  models.get -> HTTP {r.status_code}: {r.text[:300]}")
        r = httpx.post(
            f"{base}/models/{GEMINI_MODEL}:generateContent",
            headers=auth,
            json={"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}], "generationConfig": {"maxOutputTokens": 16}},
            timeout=TIMEOUT,
        )
        print(f"  generateContent -> HTTP {r.status_code}")
        show("rate-limit headers", rate_headers(r))
        if r.status_code == 429:
            try:
                details = r.json()["error"].get("details", [])
            except (ValueError, KeyError):
                details = []
            for d in details:
                for v in d.get("violations", []):
                    print(f"    quota hit: {v.get('quotaMetric')} | id={v.get('quotaId')} | value={v.get('quotaValue')}")
                if "retryDelay" in d:
                    print(f"    retryDelay: {d['retryDelay']}")
            if not details:
                print(f"    body: {json.dumps(r.json(), indent=2)[:600]}")
        elif r.status_code != 200:
            print(f"    body: {r.text[:300]}")
    except httpx.HTTPError as exc:
        print(f"  FAIL {type(exc).__name__}: {exc}")
    print("  Gemini does not return RPM/RPD/TPM in the API. Read them (per model, per project) at:")
    print("    https://aistudio.google.com/rate-limit")


def main() -> None:
    print("Probing provider limits (one tiny request per model)...")
    probe_groq()
    if "--billing" in sys.argv:  # extra ~6 requests that deliberately drain one model's TPM bucket
        probe_groq_billing()
    probe_openrouter()
    probe_gemini()


if __name__ == "__main__":
    main()

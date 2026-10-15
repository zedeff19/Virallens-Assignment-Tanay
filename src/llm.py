"""Model factory: rate limiting (per-model token bucket for Groq), retry with backoff, Groq -> OpenRouter -> Gemini fallbacks, SQLite cache.

Model assignment (verified by tests/smoke_providers.py: chat + tool-calling pass on all three):

    role              primary (Groq)            fallbacks, in order
    ----------------  ------------------------  ----------------------------------------------
    editor            openai/gpt-oss-120b       qwen/qwen3.8-27b -> OpenRouter -> gemini-2.5-flash
    analyst           qwen/qwen3.8-27b          openai/gpt-oss-120b -> OpenRouter -> gemini-2.5-flash
    retriever         openai/gpt-oss-20b        qwen/qwen3.8-27b -> OpenRouter -> gemini-2.5-flash
    verifier          openai/gpt-oss-20b        qwen/qwen3.8-27b -> OpenRouter -> gemini-2.5-flash
    classifier        openai/gpt-oss-20b        qwen/qwen3.8-27b -> OpenRouter -> gemini-2.5-flash

Groq quotas are per model, so a second Groq model is tried first. OpenRouter comes before Gemini
because Gemini's free tier is only 5 RPM / 20 RPD; OpenRouter's free models allow ~20 RPM and 50
RPD (1,000 RPD after a one-time $10 credit purchase). Groq is primary because it answered 5 rapid
calls in ~3s, whereas Gemini took ~45s for the same burst (client-side backoff on its free-tier
limit). A link is skipped when its API key is not set (OpenRouter is optional).

Groq free-tier limits (per model, from the Groq console):

    model                     RPM   RPD   TPM   TPD
    openai/gpt-oss-120b        30    1K    8K  200K
    openai/gpt-oss-20b         30    1K    8K  200K
    qwen/qwen3.8-27b           30    1K    8K  200K

Tokens, not requests, are the binding limit: a single request over 8K tokens is rejected outright,
and InMemoryRateLimiter only counts requests. DeepAgents' built-in tool schemas alone cost ~2.6K
tokens per call (measured, deepagents 0.7.24). Each Groq model therefore also gets a token bucket
(src/token_budget.py, wired in by src/budgeted_groq.py): a request is reserved at prompt estimate +
expected completion, trued up with the real usage, and waits for the bucket (or fails over once the
wait exceeds GROQ_MAX_BUDGET_WAIT) instead of earning a 429. Groq bills actual tokens, not the
requested max_tokens (verified live), so max_tokens (GROQ_MAX_COMPLETION) is only a runaway guard.
Residual TPM 429s are absorbed by retry, which now sleeps for the bucket's penalty, then fallback.

Usage:

    llm = get_llm("editor")                       # plain chat model: bind_tools / with_structured_output
    create_deep_agent(model=llm, middleware=get_agent_middleware("editor"), ...)
    clf = get_llm_with_fallbacks("classifier", lambda m: m.with_structured_output(Schema))
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

from src.token_budget import TokenBucket

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import Runnable

load_dotenv()

Role = Literal["editor", "analyst", "retriever", "verifier", "classifier"]

# Resolved against the project root so the cache is shared regardless of the working directory.
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / ".langchain.db"

# Overridable via env so a retired model name never needs a code edit.
GROQ_LARGE = os.getenv("GROQ_MODEL_LARGE", "openai/gpt-oss-120b")
GROQ_SMALL = os.getenv("GROQ_MODEL_SMALL", "openai/gpt-oss-20b")
GROQ_ALT = os.getenv("GROQ_MODEL_ALT", "qwen/qwen3.8-27b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Free OpenRouter models rotate out; this one advertises tools + structured_outputs (checked
# against https://openrouter.ai/api/v1/models and smoke-tested for tool calls and with_structured_output).
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

_ROLE_PRIMARY = {
    "editor": GROQ_LARGE,
    "analyst": GROQ_ALT,  # own 8K TPM bucket: it reads whole evidence files, so keep it off the Editor's 120b
    "retriever": GROQ_SMALL,
    "verifier": GROQ_SMALL,
    "classifier": GROQ_SMALL,
}

_KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

# Requests/second, kept just under the provider RPM (Groq 30, Gemini 5). Shared by every role. Groq
# quotas are per model, so Groq gets one limiter per model; the others one per provider.
_RPS = {"groq": 0.4, "gemini": 0.08, "openrouter": 0.3}

# Groq free-tier tokens/minute per model (see table above); GROQ_TPM overrides all of them, e.g. on a
# paid tier. GROQ_TPM_SAFETY is the fraction of it the bucket may use (estimates are approximate).
_GROQ_TPM = {GROQ_LARGE: 8000, GROQ_SMALL: 8000, GROQ_ALT: 8000}
_GROQ_TPM_DEFAULT = 8000

# Completion cap per model. max_tokens is not billed against TPM, so this only stops runaway
# generation: reasoning models spend completion tokens on thinking, keep it generous.
_MODEL_MAX_COMPLETION = {GROQ_ALT: 16384}  # qwen rejects larger values with a 400

# Sampling temperature per model. Everything is greedy (0) except Qwen: Qwen3 falls into endless
# repetition under greedy decoding (the analyst re-issued an identical write_file 8x in one trace), and
# its recommended setting is 0.6. GROQ_ALT_TEMPERATURE overrides it.
_MODEL_TEMPERATURE = {GROQ_ALT: 0.6}

# Quota exhausted for the day: waiting will not help, so skip retry and fail over immediately.
_DAILY_QUOTA_MARKERS = ("per day", "perday", "per-day", "(tpd)", "(rpd)", "daily")


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "groq" | "gemini" | "openrouter"
    model: str


def get_chain(role: Role) -> list[ModelSpec]:
    """Primary model followed by its fallbacks, skipping providers whose key is not set."""
    primary = _ROLE_PRIMARY[role]
    # Second link is a different Groq model (own quota bucket): the 120b when the primary is already qwen.
    alt = GROQ_LARGE if primary == GROQ_ALT else GROQ_ALT
    candidates = [
        ModelSpec("groq", primary),
        ModelSpec("groq", alt),
        ModelSpec("openrouter", OPENROUTER_MODEL),
        ModelSpec("gemini", GEMINI_MODEL),
    ]
    chain: list[ModelSpec] = []
    for spec in candidates:
        if spec in chain or not os.getenv(_KEY_ENV[spec.provider]):
            continue
        chain.append(spec)
    if not chain:
        raise RuntimeError(
            "No LLM provider configured: set at least one of " + ", ".join(_KEY_ENV.values())
        )
    return chain


@lru_cache(maxsize=None)
def _limiter(provider: str, model: str | None = None) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(requests_per_second=_RPS[provider], max_bucket_size=1)


@lru_cache(maxsize=None)
def _budget(model: str) -> TokenBucket:
    """Token bucket for one Groq model, shared by every role that uses it."""
    tpm = float(os.getenv("GROQ_TPM") or _GROQ_TPM.get(model, _GROQ_TPM_DEFAULT))
    return TokenBucket(
        tpm,
        safety=float(os.getenv("GROQ_TPM_SAFETY", "0.9")),
        max_wait=float(os.getenv("GROQ_MAX_BUDGET_WAIT", "15")),
        name=model,
    )


def _max_completion(model: str) -> int:
    cap = int(os.getenv("GROQ_MAX_COMPLETION", "2048"))
    return min(cap, _MODEL_MAX_COMPLETION.get(model, cap))


def _temperature(model: str) -> float:
    if model == GROQ_ALT and os.getenv("GROQ_ALT_TEMPERATURE"):
        return float(os.environ["GROQ_ALT_TEMPERATURE"])
    return _MODEL_TEMPERATURE.get(model, 0.0)


@lru_cache(maxsize=None)
def _build(provider: str, model: str) -> BaseChatModel:
    # Provider SDK retries are disabled so they do not multiply with the retry layer below.
    if provider == "groq":
        from src.budgeted_groq import BudgetedChatGroq

        return BudgetedChatGroq(
            model=model,
            temperature=_temperature(model),
            max_retries=0,
            max_tokens=_max_completion(model),
            rate_limiter=_limiter(provider, model),
            budget=_budget(model),
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=0 means "SDK default (5 retries)" for Gemini; 1 means a single attempt.
        return ChatGoogleGenerativeAI(
            model=model, temperature=0, max_retries=1, rate_limiter=_limiter(provider)
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=0,
        max_retries=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        rate_limiter=_limiter(provider),
    )


_cache_path: str | None = None


def init_cache(path: str | os.PathLike[str] | None = None) -> None:
    """Install the global SQLite LLM cache. Idempotent unless a different `path` is passed."""
    global _cache_path
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    resolved = str(path or os.getenv("LLM_CACHE_PATH") or DEFAULT_CACHE_PATH)
    if resolved == _cache_path:
        return
    set_llm_cache(SQLiteCache(database_path=resolved))
    _cache_path = resolved


def _status_code(exc: BaseException) -> int | None:
    # Provider SDKs expose it as .status_code (groq/openai) or .code (google-genai); langchain
    # wrappers keep the original error on __cause__.
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        for attr in ("status_code", "code"):
            value = getattr(cur, attr, None)
            if isinstance(value, int):
                return value
        cur = cur.__cause__ or cur.__context__
    return None


def is_retryable(exc: BaseException) -> bool:
    """True for transient failures: 429 (not daily quota), 408/409, 5xx, timeouts, connection errors.

    Everything else (400, 413 request-too-large, auth, ...) is non-retryable and goes straight to
    the next fallback, which may accept it (e.g. Gemini's 250K TPM takes what Groq's 8K TPM rejects).
    """
    # Token-budget errors (RequestTooLarge, BudgetBusy): waiting on this model cannot help.
    if getattr(exc, "retryable", None) is False:
        return False
    text = str(exc).lower()
    if any(marker in text for marker in _DAILY_QUOTA_MARKERS):
        return False
    status = _status_code(exc)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    return (
        "timeout" in name
        or "connection" in name
        or "429" in text
        or "rate limit" in text
        or "resource_exhausted" in text
    )


def get_llm(role: Role) -> BaseChatModel:
    """Primary chat model for `role` (rate-limited, temperature 0, cached)."""
    init_cache()
    spec = get_chain(role)[0]
    return _build(spec.provider, spec.model)


def get_agent_middleware(role: Role) -> list[AgentMiddleware]:
    """Retry + fallback for `create_deep_agent(middleware=...)` / `create_agent`.

    `.with_fallbacks()` returns a RunnableWithFallbacks, which has no `.bind_tools()`, so an agent
    needs the plain model from `get_llm` plus these middleware. Order matters: the first entry is
    outermost, so the fallback wraps retry and every link in the chain gets its own retries.
    """
    from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware

    init_cache()
    middleware: list[AgentMiddleware] = []
    fallbacks = [_build(s.provider, s.model) for s in get_chain(role)[1:]]
    if fallbacks:
        middleware.append(ModelFallbackMiddleware(*fallbacks))
    middleware.append(
        ModelRetryMiddleware(
            max_retries=2,
            retry_on=is_retryable,
            on_failure="error",  # re-raise so the fallback middleware takes over
            initial_delay=1.0,
            backoff_factor=2.0,
            max_delay=20.0,
        )
    )
    return middleware


def get_llm_with_fallbacks(
    role: Role, prepare: Callable[[BaseChatModel], Runnable[Any, Any]] | None = None
) -> Runnable[Any, Any]:
    """Primary model with `.with_retry()` per link, chained by `.with_fallbacks()`.

    For non-agent callers (classifier, structured output). `prepare` is applied to every link
    before wrapping, e.g. `lambda m: m.with_structured_output(Schema)` or `m.bind_tools(tools)`.
    """
    import groq
    import openai

    init_cache()
    retry_types = (
        groq.RateLimitError,
        groq.InternalServerError,
        groq.APITimeoutError,
        groq.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APITimeoutError,
        openai.APIConnectionError,
    )

    def link(spec: ModelSpec) -> Runnable[Any, Any]:
        model = _build(spec.provider, spec.model)
        runnable = prepare(model) if prepare else model
        return runnable.with_retry(
            retry_if_exception_type=retry_types, stop_after_attempt=3, wait_exponential_jitter=True
        )

    primary, *rest = (link(s) for s in get_chain(role))
    return primary.with_fallbacks(rest) if rest else primary

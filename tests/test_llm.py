"""Offline tests for src/llm.py: chain construction, retry predicate, fallback, retry, cache.

No network and no API keys needed: every model is a counting fake and `_build` is monkeypatched.
"""
from __future__ import annotations

from typing import Any

import groq
import httpx
import pytest
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src import llm


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
CALLS: list[str] = []  # names of fakes that really ran (pydantic copies list fields, so not a field)


class FakeChat(BaseChatModel):
    """Chat model that fails `fail_times` times with `error`, then answers `reply`."""

    name: str = "fake"
    reply: str = "ok"
    fail_times: int = 0
    error: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        CALLS.append(self.name)
        if CALLS.count(self.name) <= self.fail_times:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


def _rate_limit_error(message: str = "Rate limit reached ... tokens per minute (TPM)") -> Exception:
    request = httpx.Request("POST", "https://api.groq.com/x")
    response = httpx.Response(429, request=request)
    return groq.RateLimitError(message, response=response, body=None)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # Never touch the project's real .langchain.db; fakes share an llm_string so isolate per test.
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "llm_cache.db"))
    for key in (
        "GROQ_TPM",
        "GROQ_MAX_COMPLETION",
        "GROQ_MAX_BUDGET_WAIT",
        "GROQ_EXPECTED_COMPLETION",
        "GROQ_ALT_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    llm._build.cache_clear()
    llm._budget.cache_clear()
    CALLS.clear()
    yield
    set_llm_cache(None)
    llm._cache_path = None


@pytest.fixture
def all_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")


# ---------------------------------------------------------------------------
# chain construction
# ---------------------------------------------------------------------------
def test_chain_roles_and_order(all_keys):
    big = llm.get_chain("editor")
    small = llm.get_chain("retriever")
    assert [s.provider for s in big] == ["groq", "groq", "openrouter", "gemini"]
    assert big[0].model == llm.GROQ_LARGE and small[0].model == llm.GROQ_SMALL
    assert big[1].model == small[1].model == llm.GROQ_ALT
    analyst = llm.get_chain("analyst")
    assert [s.provider for s in analyst] == ["groq", "groq", "openrouter", "gemini"]
    assert analyst[0].model == llm.GROQ_ALT and analyst[1].model == llm.GROQ_LARGE
    assert llm.get_chain("verifier")[0].model == llm.GROQ_SMALL


def test_chain_skips_providers_without_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert [s.provider for s in llm.get_chain("editor")] == ["groq", "groq"]
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert [s.provider for s in llm.get_chain("editor")][-1] == "gemini"


def test_chain_without_any_key_raises():
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        llm.get_chain("editor")


def test_agent_middleware_order(all_keys):
    mw = llm.get_agent_middleware("editor")
    # fallback outermost so each link in the chain gets its own retries
    assert isinstance(mw[0], ModelFallbackMiddleware) and isinstance(mw[1], ModelRetryMiddleware)
    assert len(mw[0].models) == 3


# ---------------------------------------------------------------------------
# retry predicate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc, expected",
    [
        (_rate_limit_error(), True),
        (_rate_limit_error("Rate limit reached on tokens per day (TPD): Limit 200000"), False),
        (RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel"), False),
        (RuntimeError("429 Rate limit exceeded: free-models-per-day"), False),
        (RuntimeError("429 RESOURCE_EXHAUSTED: try again later"), True),
        (TimeoutError("timed out"), True),
        (ValueError("bad request"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert llm.is_retryable(exc) is expected


def test_token_budget_errors_are_not_retryable():
    from src.token_budget import BudgetBusy, RequestTooLarge

    assert llm.is_retryable(RequestTooLarge("too big")) is False
    assert llm.is_retryable(BudgetBusy("busy, tokens per minute")) is False


def test_is_retryable_reads_status_from_cause():
    class Sdk(Exception):
        code = 503

    try:
        try:
            raise Sdk("boom")
        except Sdk as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert llm.is_retryable(outer) is True


# ---------------------------------------------------------------------------
# fallback + retry (non-agent path)
# ---------------------------------------------------------------------------
def test_fallback_when_primary_fails_non_retryably(monkeypatch, all_keys):
    fakes = {
        ("groq", llm.GROQ_SMALL): FakeChat(name="primary", fail_times=99, error=ValueError("413")),
        ("groq", llm.GROQ_ALT): FakeChat(name="alt", reply="from-alt"),
    }
    monkeypatch.setattr(
        llm, "_build", lambda provider, model: fakes.get((provider, model)) or FakeChat(name=provider)
    )
    out = llm.get_llm_with_fallbacks("retriever").invoke("hi")
    assert out.content == "from-alt"
    assert CALLS == ["primary", "alt"]


def test_retry_recovers_from_429(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    primary = FakeChat(name="primary", reply="recovered", fail_times=1, error=_rate_limit_error())
    monkeypatch.setattr(llm, "_build", lambda provider, model: primary)
    assert llm.get_llm_with_fallbacks("retriever").invoke("hi").content == "recovered"
    assert CALLS == ["primary", "primary"]


def test_prepare_is_applied_to_each_link(monkeypatch, all_keys):
    seen: list[str] = []

    def prepare(model):
        seen.append(model.name)
        return model

    monkeypatch.setattr(llm, "_build", lambda provider, model: FakeChat(name=f"{provider}:{model}"))
    llm.get_llm_with_fallbacks("editor", prepare)
    assert len(seen) == 4


# ---------------------------------------------------------------------------
# agent middleware path
# ---------------------------------------------------------------------------
def _request(model: BaseChatModel) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[HumanMessage("hi")],
        system_message=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=None,
    )


def test_middleware_429_on_primary_falls_back(all_keys):
    primary = FakeChat(name="primary")
    alt = FakeChat(name="alt")
    attempts: list[str] = []

    def handler(request: ModelRequest) -> ModelResponse:
        attempts.append(request.model.name)
        if request.model.name == "primary":
            raise _rate_limit_error("Rate limit reached on tokens per day (TPD)")  # non-retryable
        return ModelResponse(result=[AIMessage(content="from-alt")])

    fallback = ModelFallbackMiddleware(alt)
    retry = ModelRetryMiddleware(max_retries=2, retry_on=llm.is_retryable, on_failure="error", initial_delay=0)
    response = fallback.wrap_model_call(
        _request(primary), lambda req: retry.wrap_model_call(req, handler)
    )
    assert response.result[0].content == "from-alt"
    assert attempts == ["primary", "alt"]  # daily-quota error skipped retry


def test_middleware_retries_transient_429_before_falling_back(all_keys):
    primary = FakeChat(name="primary")
    alt = FakeChat(name="alt")
    attempts: list[str] = []

    def handler(request: ModelRequest) -> ModelResponse:
        attempts.append(request.model.name)
        if request.model.name == "primary":
            raise _rate_limit_error()  # TPM: retryable
        return ModelResponse(result=[AIMessage(content="from-alt")])

    fallback = ModelFallbackMiddleware(alt)
    retry = ModelRetryMiddleware(
        max_retries=2, retry_on=llm.is_retryable, on_failure="error", initial_delay=0, jitter=False
    )
    fallback.wrap_model_call(_request(primary), lambda req: retry.wrap_model_call(req, handler))
    assert attempts == ["primary", "primary", "primary", "alt"]


# ---------------------------------------------------------------------------
# Groq token budget
# ---------------------------------------------------------------------------
class FakeCompletions:
    """Stands in for ChatGroq.client: counts calls, replays `outcomes` (exception = raise it)."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **_: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _completion(total_tokens: int) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": total_tokens - 5, "completion_tokens": 5, "total_tokens": total_tokens},
    }


def _budgeted(client: FakeCompletions, bucket) -> Any:
    from src.budgeted_groq import BudgetedChatGroq

    model = BudgetedChatGroq(model="openai/gpt-oss-20b", api_key="x", max_retries=0, budget=bucket)
    model.client = client
    return model


class _Clock:
    t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _bucket(clock: _Clock, *, max_wait: float = 60.0):
    from src.token_budget import TokenBucket

    return TokenBucket(6000, safety=1.0, max_wait=max_wait, clock=clock.now, sleep=clock.sleep)


def test_groq_build_is_budgeted_and_buckets_are_per_model(monkeypatch):
    from src.budgeted_groq import BudgetedChatGroq

    monkeypatch.setenv("GROQ_API_KEY", "x")
    small = llm._build("groq", llm.GROQ_SMALL)
    assert isinstance(small, BudgetedChatGroq)
    assert small.budget is llm._budget(llm.GROQ_SMALL)  # shared by every role on this model
    assert llm._budget(llm.GROQ_SMALL) is not llm._budget(llm.GROQ_ALT)
    assert llm._limiter("groq", llm.GROQ_SMALL) is not llm._limiter("groq", llm.GROQ_ALT)
    assert small.max_tokens == 2048


def test_max_completion_is_a_clamped_guard(monkeypatch):
    assert llm._max_completion(llm.GROQ_SMALL) == 2048
    monkeypatch.setenv("GROQ_MAX_COMPLETION", "20000")
    assert llm._max_completion(llm.GROQ_SMALL) == 20000
    assert llm._max_completion(llm.GROQ_ALT) == 16384  # qwen rejects more with a 400


def test_qwen_is_not_greedy_but_other_models_are(monkeypatch):
    # greedy decoding made Qwen repeat an identical tool call 8x in a real trace
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm._build("groq", llm.GROQ_ALT).temperature == 0.6
    # ChatGroq turns 0 into 1e-8 because Groq rejects exactly 0
    assert llm._build("groq", llm.GROQ_SMALL).temperature < 1e-6
    assert llm._build("groq", llm.GROQ_LARGE).temperature < 1e-6


def test_qwen_temperature_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GROQ_ALT_TEMPERATURE", "0.3")
    assert llm._temperature(llm.GROQ_ALT) == 0.3
    assert llm._temperature(llm.GROQ_SMALL) == 0  # the override only targets the alt model


def test_budgeted_call_is_trued_up_with_real_usage():
    clock = _Clock()
    bucket = _bucket(clock)
    model = _budgeted(FakeCompletions(_completion(1500)), bucket)
    assert model.invoke("hi").content == "hi"
    assert bucket.level == pytest.approx(6000 - 1500)  # estimate refunded, real usage charged


def test_budgeted_overshoot_is_charged_and_next_call_waits():
    clock = _Clock()
    bucket = _bucket(clock)
    model = _budgeted(FakeCompletions(_completion(9000), _completion(100)), bucket)
    model.invoke("hi")
    assert bucket.level == pytest.approx(-3000)
    model.invoke("hi again")
    assert clock.t > 25  # waited out (most of) the 3000-token debt at 100 tokens/s


def test_budgeted_tpm_429_penalizes_by_retry_after():
    clock = _Clock()
    bucket = _bucket(clock)
    err = _rate_limit_error("Rate limit reached ... tokens per minute (TPM) ... Please try again in 5s.")
    client = FakeCompletions(err, _completion(100))
    model = _budgeted(client, bucket)
    with pytest.raises(groq.RateLimitError):
        model.invoke("hi")
    model.invoke("hi")  # the retry layer calls again: the bucket makes it wait out the 5s
    assert clock.t == pytest.approx(5.0)
    assert client.calls == 2


def test_budgeted_other_failure_refunds_the_reservation():
    clock = _Clock()
    bucket = _bucket(clock)
    model = _budgeted(FakeCompletions(ValueError("400 bad request")), bucket)
    with pytest.raises(ValueError):
        model.invoke("hi")
    assert bucket.level == pytest.approx(6000)


def test_oversize_request_never_reaches_groq_and_chain_falls_over(monkeypatch, all_keys):
    from src.token_budget import RequestTooLarge, TokenBucket

    tiny = TokenBucket(1000, safety=1.0)  # capacity 1000 < 450 allowance + ~1700-token prompt
    groq_client = FakeCompletions()
    primary = _budgeted(groq_client, tiny)
    with pytest.raises(RequestTooLarge):
        primary.invoke("x" * 5000)
    assert groq_client.calls == 0

    fakes = {("groq", llm.GROQ_SMALL): primary, ("groq", llm.GROQ_ALT): FakeChat(name="alt", reply="from-alt")}
    monkeypatch.setattr(
        llm, "_build", lambda provider, model: fakes.get((provider, model)) or FakeChat(name=provider)
    )
    out = llm.get_llm_with_fallbacks("retriever").invoke("x" * 5000)
    assert out.content == "from-alt"
    assert groq_client.calls == 0 and CALLS == ["alt"]


def test_estimate_ignores_max_tokens(monkeypatch):
    from src.budgeted_groq import BudgetedChatGroq

    messages = [HumanMessage("hello " * 200)]
    small = BudgetedChatGroq(model="m", api_key="x", max_tokens=16)
    large = BudgetedChatGroq(model="m", api_key="x", max_tokens=20000)
    assert small._estimate(messages, None, {}) == large._estimate(messages, None, {})


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def test_cache_hit_makes_no_second_call(tmp_path):
    llm.init_cache(tmp_path / "cache.db")
    model = FakeChat(name="m")
    assert model.invoke("same question").content == "ok"
    assert model.invoke("same question").content == "ok"
    assert CALLS == ["m"]
    assert (tmp_path / "cache.db").exists()


def test_init_cache_is_idempotent(tmp_path):
    path = tmp_path / "cache.db"
    llm.init_cache(path)
    from langchain_core.globals import get_llm_cache

    first = get_llm_cache()
    llm.init_cache(path)
    assert get_llm_cache() is first

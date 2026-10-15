"""Offline tests for src/token_budget.py: bucket arithmetic, estimator, retry-after parsing.

A fake clock drives the bucket, so nothing sleeps for real. Numbers are chosen to be round:
tpm=6000, safety=1.0 -> capacity 6000 tokens, refill 100 tokens/s.
"""
from __future__ import annotations

import asyncio
import threading

import groq
import httpx
import pytest

from src import token_budget as tb
from src.token_budget import BudgetBusy, RequestTooLarge, TokenBucket


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def make(clock: Clock, *, max_wait: float = 60.0, tpm: float = 6000) -> TokenBucket:
    return TokenBucket(tpm, safety=1.0, max_wait=max_wait, clock=clock.now, sleep=clock.sleep)


def groq_429(message: str, headers: dict[str, str] | None = None) -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/x")
    response = httpx.Response(429, request=request, headers=headers or {})
    return groq.RateLimitError(message, response=response, body=None)


# ---------------------------------------------------------------------------
# bucket
# ---------------------------------------------------------------------------
def test_take_within_capacity_does_not_sleep():
    clock = Clock()
    bucket = make(clock)
    bucket.reserve(2500)
    bucket.reserve(2500)
    assert clock.slept == []
    assert bucket.level == pytest.approx(1000)


def test_deficit_sleeps_deficit_over_rate():
    clock = Clock()
    bucket = make(clock)
    bucket.reserve(4000)  # level 2000
    bucket.reserve(4000)  # needs 2000 more at 100/s
    assert clock.slept == [pytest.approx(20.0)]


def test_refill_is_capped_at_capacity():
    clock = Clock()
    bucket = make(clock)
    bucket.reserve(1000)
    clock.t += 10_000
    assert bucket.level == pytest.approx(6000)


def test_release_refunds_in_full():
    bucket = make(Clock())
    res = bucket.reserve(3000)
    bucket.release(res)
    assert bucket.level == pytest.approx(6000)


def test_settle_refunds_an_overestimate():
    bucket = make(Clock())
    res = bucket.reserve(3000)  # level 3000
    bucket.settle(res, 1000)
    assert bucket.level == pytest.approx(5000)


def test_settle_charges_an_overshoot_and_next_caller_waits_the_debt():
    clock = Clock()
    bucket = make(clock)
    res = bucket.reserve(1000)  # level 5000
    bucket.settle(res, 8000)  # a reasoning completion ran long: level 5000 + 1000 - 8000
    assert bucket.level == pytest.approx(-2000)
    bucket.reserve(500)  # must repay 2000 and then cover 500
    assert clock.slept == [pytest.approx(25.0)]


def test_penalize_makes_the_next_reserve_wait_retry_after():
    clock = Clock()
    bucket = make(clock)
    res = bucket.reserve(3000)
    bucket.penalize(res, retry_after=5.0)
    bucket.reserve(3000)
    assert clock.slept == [pytest.approx(5.0)]


def test_request_that_can_never_fit_raises_without_sleeping():
    clock = Clock()
    bucket = make(clock)
    with pytest.raises(RequestTooLarge) as info:
        bucket.reserve(6001)
    assert info.value.status_code == 413 and info.value.retryable is False
    assert clock.slept == []


def test_wait_beyond_max_wait_raises_budget_busy():
    clock = Clock()
    bucket = make(clock, max_wait=10)
    bucket.reserve(4000)  # level 2000, the next 4000 needs 20s
    with pytest.raises(BudgetBusy) as info:
        bucket.reserve(4000)
    assert info.value.retryable is False
    assert clock.slept == []


def test_daily_quota_retry_after_fails_fast_instead_of_sleeping():
    clock = Clock()
    bucket = make(clock, max_wait=15)
    res = bucket.reserve(3000)
    bucket.penalize(res, retry_after=23 * 60)  # "try again in 23m"
    with pytest.raises(BudgetBusy):
        bucket.reserve(3000)


def test_async_reserve_waits_on_the_async_sleep():
    clock = Clock()
    slept: list[float] = []

    async def asleep(seconds: float) -> None:
        slept.append(seconds)
        clock.t += seconds

    async def go() -> None:
        bucket = TokenBucket(6000, safety=1.0, max_wait=60, clock=clock.now, asleep=asleep)
        await bucket.areserve(4000)
        await bucket.areserve(4000)

    asyncio.run(go())
    assert slept == [pytest.approx(20.0)]


def test_threads_cannot_overdraw_the_bucket():
    # capacity 1000, refill 1/60 token/s (nothing refills during the test), max_wait=0 so a
    # request that does not fit raises instead of waiting. 20 threads x 100 tokens -> exactly 10 fit.
    bucket = TokenBucket(1, safety=1000, max_wait=0)
    granted: list[int] = []
    refused: list[int] = []
    gate = threading.Barrier(20)

    def worker() -> None:
        gate.wait()
        try:
            bucket.reserve(100)
            granted.append(1)
        except BudgetBusy:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (len(granted), len(refused)) == (10, 10)


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------
def _prompt(chars: int) -> list[dict]:
    return [{"role": "user", "content": "x" * chars}]


def test_estimate_is_prompt_over_chars_per_token_plus_allowance():
    est = tb.estimate_tokens(_prompt(3000), chars_per_token=3.0, expected_completion=450)
    # payload is the JSON of [messages, tools]: a few dozen chars of framing around the 3000
    assert 1000 + 450 <= est <= 1030 + 450


def test_estimate_counts_tool_schemas():
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 3000}}]
    assert tb.estimate_tokens(_prompt(10), tools) > tb.estimate_tokens(_prompt(10)) + 900


def test_env_overrides_the_completion_allowance(monkeypatch):
    base = tb.estimate_tokens(_prompt(10))
    monkeypatch.setenv("GROQ_EXPECTED_COMPLETION", "1000")
    assert tb.estimate_tokens(_prompt(10)) == base - 450 + 1000


def test_5_5k_prompt_fits_an_8k_tpm_model():
    # 0.9 x 8000 = 7200 capacity; ~5.5K prompt + 450 allowance must NOT be rejected
    bucket = TokenBucket(8000)
    est = tb.estimate_tokens(_prompt(5500 * 3), chars_per_token=3.0, expected_completion=450)
    assert 5900 <= est <= 6100
    bucket.reserve(est)  # no RequestTooLarge, no BudgetBusy


def test_prompt_above_capacity_minus_allowance_is_too_large():
    bucket = TokenBucket(8000)
    est = tb.estimate_tokens(_prompt(7000 * 3), chars_per_token=3.0, expected_completion=450)
    assert est > 7200
    with pytest.raises(RequestTooLarge):
        bucket.reserve(est)


# ---------------------------------------------------------------------------
# retry-after
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message, expected",
    [
        ("Rate limit reached ... Limit 8000, Used 7000, Requested 3000. Please try again in 3.2s.", 3.2),
        ("tokens per minute (TPM) ... Please try again in 1m2.4s. Visit ...", 62.4),
        ("Please try again in 850ms.", 0.85),
        ("tokens per day (TPD) ... Please try again in 23m4.5s.", 23 * 60 + 4.5),
        ("Please try again in 1h2m3s.", 3723.0),
    ],
)
def test_parse_retry_after_from_message(message, expected):
    assert tb.parse_retry_after(groq_429(message)) == pytest.approx(expected)


def test_parse_retry_after_falls_back_to_header():
    assert tb.parse_retry_after(groq_429("Rate limit reached", {"retry-after": "7"})) == 7.0


def test_parse_retry_after_none_when_no_hint():
    assert tb.parse_retry_after(groq_429("Rate limit reached")) is None
    assert tb.parse_retry_after(RuntimeError("boom")) is None

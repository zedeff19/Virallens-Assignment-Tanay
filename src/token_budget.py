"""Per-model token bucket for Groq's tokens-per-minute (TPM) quota.

`BaseChatModel.rate_limiter` never sees the messages, so it cannot count tokens. This module is the
provider-free core (bucket, estimator, retry-after parser); `src/budgeted_groq.py` wires it into
ChatGroq.

Groq bills ACTUAL tokens (prompt + real completion) against TPM, not the requested max_tokens
(verified live), so a request is reserved at `prompt estimate + expected completion` and trued up
with the real usage once the response arrives. An overshoot drives the bucket negative and the
next caller waits out the debt.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# Seconds to back off when a 429 carries no usable retry hint.
DEFAULT_PENALTY = 10.0


class _BudgetError(Exception):
    # Read by llm.is_retryable and ignored by with_retry: waiting on the same model cannot help,
    # so the next link in the fallback chain is tried immediately.
    retryable = False


class RequestTooLarge(_BudgetError):
    """The request can never fit the model's bucket; raised before any network call."""

    status_code = 413


class BudgetBusy(_BudgetError):
    """The bucket needs longer than `max_wait` to cover the request; fail over instead of waiting."""


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def estimate_tokens(
    message_dicts: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    chars_per_token: float | None = None,
    expected_completion: int | None = None,
) -> int:
    """Prompt tokens (messages + tool schemas / chars_per_token) plus a modest completion allowance.

    Deliberately ignores max_tokens: Groq does not bill it. The allowance is only the reservation
    placeholder; `TokenBucket.settle` replaces it with the real figure.
    """
    cpt = chars_per_token or _env_number("GROQ_CHARS_PER_TOKEN", 3.0)
    completion = (
        expected_completion
        if expected_completion is not None
        else int(_env_number("GROQ_EXPECTED_COMPLETION", 450))
    )
    payload = json.dumps([message_dicts, tools or []], default=str, ensure_ascii=False)
    return math.ceil(len(payload) / cpt) + completion


_RETRY_HINT = re.compile(r"try again in\s+([0-9hms.\s]+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_retry_after(exc: BaseException) -> float | None:
    """Seconds Groq asked us to wait: 'try again in 1m2.4s' / '3.2s' / '850ms', else Retry-After."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        match = _RETRY_HINT.search(str(cur))
        if match:
            parts = _DURATION_PART.findall(match.group(1))
            if parts:
                return sum(float(n) * _UNIT_SECONDS[unit] for n, unit in parts)
        headers = getattr(getattr(cur, "response", None), "headers", None)
        if headers is not None:
            try:
                return float(headers.get("retry-after"))
            except (TypeError, ValueError):
                pass
        cur = cur.__cause__ or cur.__context__
    return None


@dataclass
class Reservation:
    tokens: int


class TokenBucket:
    """Continuously refilling bucket: capacity = tpm * safety, refill = tpm / 60 per second.

    `level` may go negative (debt from a completion longer than estimated); `reserve` then waits
    until the debt is repaid and the request fits.
    """

    def __init__(
        self,
        tpm: float,
        *,
        safety: float = 0.9,
        max_wait: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        name: str = "",
    ) -> None:
        self.capacity = tpm * safety
        self.rate = tpm / 60.0
        self.max_wait = max_wait
        self.name = name
        self._clock = clock
        self._sleep = sleep
        self._asleep = asleep
        self._lock = threading.Lock()
        self._level = self.capacity
        self._last = clock()

    @property
    def level(self) -> float:
        with self._lock:
            self._refill()
            return self._level

    def _refill(self) -> None:  # caller holds the lock
        now = self._clock()
        self._level = min(self.capacity, self._level + (now - self._last) * self.rate)
        self._last = now

    def _step(self, est: int) -> float:
        """Take `est` tokens and return 0.0, or return the seconds to wait. Raises when hopeless."""
        if est > self.capacity:
            raise RequestTooLarge(
                f"{self.name or 'model'}: request needs ~{est} tokens, bucket holds {self.capacity:.0f}"
            )
        with self._lock:
            self._refill()
            if self._level >= est:
                self._level -= est
                return 0.0
            wait = (est - self._level) / self.rate
        if wait > self.max_wait:
            raise BudgetBusy(
                f"{self.name or 'model'}: token budget needs {wait:.1f}s (max_wait {self.max_wait:.0f}s)"
            )
        return wait

    def reserve(self, est: int) -> Reservation:
        while (wait := self._step(est)) > 0:
            self._sleep(wait)
        return Reservation(est)

    async def areserve(self, est: int) -> Reservation:
        while (wait := self._step(est)) > 0:
            await self._asleep(wait)
        return Reservation(est)

    def settle(self, res: Reservation, actual: int) -> None:
        """True up with the real total_tokens: refund an overestimate, charge an overshoot."""
        with self._lock:
            self._refill()
            self._level = min(self.capacity, self._level + res.tokens - actual)

    def release(self, res: Reservation) -> None:
        """Full refund: the request failed without consuming tokens."""
        with self._lock:
            self._refill()
            self._level = min(self.capacity, self._level + res.tokens)

    def penalize(self, res: Reservation, retry_after: float) -> None:
        """Groq rejected the request: set the bucket so the same request fits after `retry_after`s."""
        with self._lock:
            self._refill()
            self._level = res.tokens - self.rate * retry_after

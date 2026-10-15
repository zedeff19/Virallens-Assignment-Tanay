"""ChatGroq that reserves tokens from a per-model `TokenBucket` before every request.

The hooks are `_generate/_agenerate/_stream/_astream`, which langchain calls after the cache lookup,
so SQLite cache hits cost no budget. Bound tools arrive in `kwargs["tools"]`, so tool-schema tokens
are part of the estimate.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import groq
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_groq import ChatGroq
from pydantic import Field

from src.token_budget import DEFAULT_PENALTY, Reservation, TokenBucket, estimate_tokens, parse_retry_after


def _total_tokens(result: ChatResult) -> int | None:
    usage = (result.llm_output or {}).get("token_usage") or {}
    total = usage.get("total_tokens")
    return int(total) if isinstance(total, (int, float)) else None


class BudgetedChatGroq(ChatGroq):
    budget: Any = Field(default=None, exclude=True)  # TokenBucket; Any keeps pydantic out of it

    def _estimate(self, messages: list[BaseMessage], stop: list[str] | None, kwargs: dict[str, Any]) -> int:
        message_dicts, _ = self._create_message_dicts(messages, stop)
        return estimate_tokens(message_dicts, kwargs.get("tools"))

    def _on_error(self, res: Reservation, exc: Exception) -> None:
        bucket: TokenBucket = self.budget
        if isinstance(exc, groq.RateLimitError):
            retry_after = parse_retry_after(exc)
            bucket.penalize(res, DEFAULT_PENALTY if retry_after is None else retry_after)
        else:
            bucket.release(res)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.budget is None or self.streaming:  # streaming reserves once, in _stream
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        res = self.budget.reserve(self._estimate(messages, stop, kwargs))
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            self._on_error(res, exc)
            raise
        actual = _total_tokens(result)
        self.budget.settle(res, res.tokens if actual is None else actual)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.budget is None or self.streaming:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        res = await self.budget.areserve(self._estimate(messages, stop, kwargs))
        try:
            result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            self._on_error(res, exc)
            raise
        actual = _total_tokens(result)
        self.budget.settle(res, res.tokens if actual is None else actual)
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.budget is None:
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return
        res = self.budget.reserve(self._estimate(messages, stop, kwargs))
        actual: int | None = None
        failed = False
        try:
            for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                usage = getattr(chunk.message, "usage_metadata", None)
                if usage and usage.get("total_tokens") is not None:
                    actual = usage["total_tokens"]
                yield chunk
        except Exception as exc:
            failed = True
            self._on_error(res, exc)
            raise
        finally:
            # Also runs on early close (GeneratorExit): without usage data the estimate stands.
            if not failed:
                self.budget.settle(res, res.tokens if actual is None else actual)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.budget is None:
            async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
            return
        res = await self.budget.areserve(self._estimate(messages, stop, kwargs))
        actual: int | None = None
        failed = False
        try:
            async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                usage = getattr(chunk.message, "usage_metadata", None)
                if usage and usage.get("total_tokens") is not None:
                    actual = usage["total_tokens"]
                yield chunk
        except Exception as exc:
            failed = True
            self._on_error(res, exc)
            raise
        finally:
            if not failed:
                self.budget.settle(res, res.tokens if actual is None else actual)

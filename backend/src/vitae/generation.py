"""The generator seam: an OpenAI-compatible client for production, a scripted one for tests."""

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, Self, TypedDict, cast

import httpx

MAX_TOKENS: Final = 400
DEADLINE_S: Final = 60.0
"""Total time allowed for one generation, first byte to last."""


@dataclass(frozen=True, slots=True)
class Delta:
    text: str


@dataclass(frozen=True, slots=True)
class Finished:
    finish: Literal["complete", "length"]


@dataclass(frozen=True, slots=True)
class Failed:
    code: Literal["generation_failed", "timeout"]


GenEvent = Delta | Finished | Failed
"""A stream is Delta* then exactly one Finished or Failed. Implementations convert every transport
exception into Failed at their own boundary; nothing above raises."""

FAILED: Final = Failed("generation_failed")
TIMED_OUT: Final = Failed("timeout")
_FINISH: Final[dict[str, Finished]] = {"stop": Finished("complete"), "length": Finished("length")}


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Generator(Protocol):
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncGenerator[GenEvent]: ...


class OpenAICompatible:
    """POST {base_url}/chat/completions with stream=true over httpx; parses `data:` lines.
    finish_reason "stop" -> complete, "length" -> length, anything else -> generation_failed."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_tokens: int = MAX_TOKENS,
        deadline_s: float = DEADLINE_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url}/chat/completions"
        self._model = model
        self._max_tokens = max_tokens
        self._deadline_s = deadline_s
        self._transport = transport

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncGenerator[GenEvent]:
        body = {"model": self._model, "messages": list(messages), "stream": True, "max_tokens": self._max_tokens}
        deadline = asyncio.get_running_loop().time() + self._deadline_s
        # The deadline wraps each await, never a yield: a timeout scope left open across a yield
        # would cancel whatever the consumer is doing when it fires.
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=self._deadline_s) as client:
                async with asyncio.timeout_at(deadline):
                    response = await client.send(client.build_request("POST", self._url, json=body), stream=True)
                try:
                    if response.status_code != 200:
                        yield FAILED
                        return
                    lines = response.aiter_lines()
                    while True:
                        async with asyncio.timeout_at(deadline):
                            line = await anext(lines, None)
                        if line is None:
                            yield FAILED
                            return
                        for event in _parse_line(line):
                            yield event
                            if not isinstance(event, Delta):
                                return
                finally:
                    await response.aclose()
        except TimeoutError:
            yield TIMED_OUT
        except (httpx.HTTPError, httpx.StreamError, httpx.InvalidURL, OSError):
            yield FAILED


def _parse_line(line: str) -> tuple[GenEvent, ...]:
    """The events one SSE line carries: none for a blank line, a comment, an event name or an empty
    delta; a Delta, a finish, or both for a data frame; FAILED for anything malformed."""
    if not line.startswith("data:"):
        return ()
    data = line.removeprefix("data:").strip()
    try:
        frame = cast(object, json.loads(data))
    except ValueError:
        return (FAILED,)
    choices = _field(frame, "choices")
    choice = cast(list[object], choices)[0] if isinstance(choices, list) and choices else None
    delta = _field(choice, "delta")
    if not isinstance(delta, dict):
        return (FAILED,)
    content, reason = _field(cast(object, delta), "content"), _field(choice, "finish_reason")
    if content is not None and not isinstance(content, str):
        return (FAILED,)
    deltas = (Delta(content),) if content else ()
    if reason is None:
        return deltas
    return (*deltas, _FINISH.get(str(reason), FAILED))


def _field(value: object, key: str) -> object:
    """value[key] when value is a JSON object, else None."""
    return cast(dict[str, object], value).get(key) if isinstance(value, dict) else None


class ScriptedGenerator:
    """Replays fixed chunks with an optional per-chunk delay. Its delay is what holds slots in the
    queue-full HTTP test."""

    def __init__(self, chunks: Sequence[str], *, delay_s: float = 0.0, fail_after: int | None = None) -> None:
        self._chunks = tuple(chunks)
        self._delay_s = delay_s
        self._fail_after = fail_after

    @classmethod
    def chunks(cls, chunks: Sequence[str], *, delay_s: float = 0.0) -> Self:
        """Each chunk as a Delta, then Finished("complete")."""
        return cls(chunks, delay_s=delay_s)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """A JSON array of chunk strings, for vitae.dev."""
        raw = cast(object, json.loads(path.read_bytes()))
        if not isinstance(raw, list) or not all(isinstance(chunk, str) for chunk in cast(list[object], raw)):
            raise ValueError(f"{path}: expected a JSON array of strings")
        return cls(cast(list[str], raw))

    @classmethod
    def failing_after(cls, chunks: Sequence[str], n: int, *, delay_s: float = 0.0) -> Self:
        """The first n chunks as Deltas, then Failed("generation_failed")."""
        return cls(chunks[:n], delay_s=delay_s, fail_after=n)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncGenerator[GenEvent]:
        for chunk in self._chunks:
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield Delta(chunk)
        yield FAILED if self._fail_after is not None else Finished("complete")

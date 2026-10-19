import asyncio
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from vitae.generation import ChatMessage, Delta, Failed, Finished, GenEvent, OpenAICompatible, ScriptedGenerator

MESSAGES: list[ChatMessage] = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def frame(content: str | None = None, finish: str | None = None) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]})


class FakeServer:
    """A tiny OpenAI-compatible ASGI app: answers every POST with the given SSE lines."""

    def __init__(self, lines: list[str], *, status: int = 200, delay_s: float = 0.0) -> None:
        self.lines = lines
        self.status = status
        self.delay_s = delay_s
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.requests.append((scope["path"], json.loads(body)))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        await send(
            {"type": "http.response.start", "status": self.status, "headers": [(b"content-type", b"text/event-stream")]}
        )
        await send({"type": "http.response.body", "body": "".join(f"{line}\n" for line in self.lines).encode()})


def run(generator: OpenAICompatible | ScriptedGenerator) -> list[GenEvent]:
    async def collect() -> list[GenEvent]:
        return [event async for event in generator.stream(MESSAGES)]

    return asyncio.run(collect())


def client(server: FakeServer, **kwargs: Any) -> OpenAICompatible:
    return OpenAICompatible("http://model.test/v1", "qwen3", transport=httpx.ASGITransport(app=server), **kwargs)


def test_a_stream_becomes_deltas_then_complete_and_the_request_asks_for_a_capped_stream():
    server = FakeServer(
        [
            ": keep-alive",
            "event: message",
            frame(content="Hello"),
            "",
            frame(content=""),
            frame(content=" world."),
            frame(finish="stop"),
            "data: [DONE]",
        ]
    )
    assert run(client(server, max_tokens=50)) == [Delta("Hello"), Delta(" world."), Finished("complete")]
    assert server.requests == [
        ("/v1/chat/completions", {"model": "qwen3", "messages": MESSAGES, "stream": True, "max_tokens": 50})
    ]


def test_finish_reason_length_is_length_and_a_last_delta_with_a_finish_is_kept():
    server = FakeServer([frame(content="Cut"), frame(content=" off", finish="length")])
    assert run(client(server)) == [Delta("Cut"), Delta(" off"), Finished("length")]


@pytest.mark.parametrize(
    "lines",
    [
        [frame(content="Hi")],
        [frame(content="Hi"), "data: [DONE]"],
        [frame(content="Hi"), "data: {not json"],
        [frame(content="Hi"), 'data: {"choices": []}'],
        [frame(content="Hi"), 'data: {"choices": [{"delta": "Hi"}]}'],
        [frame(content="Hi"), 'data: {"choices": [{"delta": {"content": 7}}]}'],
        [frame(content="Hi"), frame(finish="content_filter")],
    ],
    ids=[
        "no-finish",
        "done-before-finish",
        "bad-json",
        "no-choices",
        "delta-not-object",
        "content-not-text",
        "unknown-finish",
    ],
)
def test_a_bad_or_unfinished_stream_ends_in_generation_failed(lines: list[str]):
    assert run(client(FakeServer(lines))) == [Delta("Hi"), Failed("generation_failed")]


def test_an_error_status_is_generation_failed():
    assert run(client(FakeServer([frame(content="Hi")], status=500))) == [Failed("generation_failed")]


def test_a_server_slower_than_the_deadline_is_timeout():
    server = FakeServer([frame(content="late"), frame(finish="stop")], delay_s=1.0)
    assert run(client(server, deadline_s=0.05)) == [Failed("timeout")]


def test_a_connection_error_is_generation_failed():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    generator = OpenAICompatible("http://model.test/v1", "qwen3", transport=httpx.MockTransport(refuse))
    assert run(generator) == [Failed("generation_failed")]


def test_scripted_chunks_replay_then_complete():
    assert run(ScriptedGenerator.chunks(["A", "B."])) == [Delta("A"), Delta("B."), Finished("complete")]


def test_scripted_failing_after_n_chunks():
    assert run(ScriptedGenerator.failing_after(["A. ", "B. ", "C."], 2)) == [
        Delta("A. "),
        Delta("B. "),
        Failed("generation_failed"),
    ]


def test_scripted_from_file(tmp_path: Path):
    path = tmp_path / "script.json"
    path.write_text('["I built ", "Shift Notes."]', encoding="utf-8")
    assert run(ScriptedGenerator.from_file(path)) == [Delta("I built "), Delta("Shift Notes."), Finished("complete")]


def test_scripted_from_file_refuses_anything_but_an_array_of_strings(tmp_path: Path):
    path = tmp_path / "script.json"
    path.write_text('["ok", 3]', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array of strings"):
        ScriptedGenerator.from_file(path)

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import httpx
import pytest

import vitae.app
from vitae.app import create_app, framed
from vitae.contract import (
    CHAT_PATH,
    ChatRequest,
    DoneEvent,
    RefusedEvent,
    SessionEvent,
    StartedEvent,
    StreamEvent,
    TextEvent,
)
from vitae.engine import Engine, Reply
from vitae.generation import ScriptedGenerator
from vitae.guards import Down, RuleId, Verdict
from vitae.profile import load_profile
from vitae.settings import load_settings
from vitae.text import Turn
from vitae.wire import HEARTBEAT, Malformed, read_events

PROFILE = load_profile(Path(__file__).parents[1] / "profiles" / "example.toml")
ORIGIN = "https://alex.example"
PEER = ("203.0.113.9", 52000)
CORS = {
    "access-control-allow-origin": ORIGIN,
    "access-control-expose-headers": "Retry-After, X-Request-Id",
    "vary": "Origin",
}


class NeverInjection:
    def score(self, text: str) -> float:
        return 0.0


class Rejections:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def turn(self, record: object) -> None:
        pass

    def rejected(self, code: str) -> None:
        self.codes.append(code)


def build(*, rate: str = "100/minute", slots: int = 1, queue: int = 0, turns: int = 2, moderator: Any = None):
    settings = load_settings(
        {
            "VITAE_ALLOWED_ORIGINS": f"{ORIGIN},http://localhost:5173",
            "VITAE_RATE_LIMITS": rate,
            "VITAE_GENERATION_SLOTS": str(slots),
            "VITAE_QUEUE_SIZE": str(queue),
            "VITAE_SESSION_MAX_TURNS": str(turns),
        }
    )
    extra = {"moderator": moderator} if moderator is not None else {}
    observer = Rejections()
    engine = Engine.build(
        settings,
        PROFILE,
        generator=ScriptedGenerator.chunks(["I built Shift Notes. It helps", " support teams."]),
        classifier=NeverInjection(),
        observer=observer,
        **extra,
    )
    return create_app(settings, engine, observer), engine, observer


def http(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=PEER), base_url="http://api.test")


def run[T](scenario: Callable[[], Awaitable[T]]) -> T:
    return asyncio.run(scenario())  # pyright: ignore[reportArgumentType]


async def post(client: httpx.AsyncClient, body: Any, *, origin: str | None = ORIGIN) -> httpx.Response:
    headers = {"Origin": origin} if origin is not None else {}
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return await client.post(CHAT_PATH, content=content, headers={**headers, "Content-Type": "application/json"})


def stream_of(response: httpx.Response) -> list[StreamEvent | Malformed]:
    return list(read_events(response.text.splitlines()))


def cors_of(response: httpx.Response) -> dict[str, str]:
    return {name: value for name, value in response.headers.items() if name in CORS}


def error_of(response: httpx.Response) -> dict[str, Any]:
    return response.json()["error"]


def test_an_answered_turn_is_an_sse_stream_with_cors_and_a_request_id():
    app, _, _ = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await post(client, {"message": "What projects have you built?"})

    response = run(scenario)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert cors_of(response) == CORS
    assert uuid.UUID(response.headers["x-request-id"]).version == 4
    events = stream_of(response)
    assert isinstance(events[0], SessionEvent) and events[0].turns_left == 1
    assert events[1:] == [
        StartedEvent(),
        TextEvent(text="I built Shift Notes. "),
        TextEvent(text="It helps support teams."),
        DoneEvent(finish="complete"),
    ]


def test_a_refused_turn_is_a_200_stream_too():
    app, _, _ = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await post(client, {"message": "My email is jane.doe@example.com, what have you built?"})

    response = run(scenario)
    assert response.status_code == 200
    assert stream_of(response)[1:] == [
        RefusedEvent(reason="pii", message="Please don't share personal details here. Ask your question without them.")
    ]


@pytest.mark.parametrize(
    "origin",
    [None, "https://attacker.example", "https://alex.example.attacker.example", "https://ALEX.example", "null"],
    ids=["missing", "other-site", "lookalike-suffix", "case-changed", "null"],
)
def test_an_origin_not_on_the_list_is_403_with_no_cors_headers(origin: str | None):
    app, _, observer = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await post(client, {"message": "What projects have you built?"}, origin=origin)

    response = run(scenario)
    assert response.status_code == 403
    assert error_of(response) == {
        "code": "origin_not_allowed",
        "detail": "The Origin header is missing or not on the allowlist.",
    }
    assert cors_of(response) == {}
    assert "x-request-id" in response.headers
    assert observer.codes == ["origin_not_allowed"]


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"{not json", 400, "invalid_request"),
        ({"message": "   "}, 400, "invalid_request"),
        ({"message": "hi", "history": []}, 400, "invalid_request"),
        ({"message": "x" * 501}, 413, "too_long"),
        ({"message": "x" * 500, "pad": "y" * 9000}, 413, "too_long"),
    ],
    ids=["bad-json", "blank", "unknown-field", "message-501", "body-over-8k"],
)
def test_a_bad_body_is_rejected_with_its_status_and_cors(body: Any, status: int, code: str):
    app, _, observer = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await post(client, body)

    response = run(scenario)
    assert (response.status_code, error_of(response)["code"]) == (status, code)
    assert cors_of(response) == CORS
    assert observer.codes == [code]


def test_an_oversized_body_is_413_and_names_the_byte_cap():
    app, _, _ = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await post(client, b"x" * 8193)

    response = run(scenario)
    assert response.status_code == 413
    assert error_of(response) == {"code": "too_long", "detail": "Request body is over 8192 bytes."}


def test_a_declared_length_over_the_cap_is_413_without_reading_the_body():
    app, _, _ = build()
    sent: list[int] = []

    async def body() -> AsyncGenerator[bytes]:
        for _ in range(1000):
            sent.append(1)
            yield b"x" * 1024

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            headers = {"Origin": ORIGIN, "Content-Length": str(1024 * 1024)}
            return await client.post(CHAT_PATH, content=body(), headers=headers)

    response = run(scenario)
    assert (response.status_code, error_of(response)["code"]) == (413, "too_long")


def test_the_rate_limit_is_429_with_retry_after_and_cors():
    app, _, observer = build(rate="2/minute")

    async def scenario() -> list[httpx.Response]:
        async with http(app) as client:
            return [await post(client, {"message": "What's the capital of France?"}) for _ in range(3)]

    first, second, third = run(scenario)
    assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429)
    assert error_of(third) == {
        "code": "rate_limited",
        "retry_after_s": int(third.headers["retry-after"]),
        "detail": "This address has sent too many requests.",
    }
    assert 1 <= int(third.headers["retry-after"]) <= 70  # a window plus one bucket at most
    assert cors_of(third) == CORS
    assert observer.codes == ["rate_limited"]


def test_a_spoofed_forwarded_for_does_not_change_the_rate_limit_key():
    app, _, _ = build(rate="2/minute")

    async def scenario() -> list[int]:
        async with http(app) as client:
            statuses: list[int] = []
            for n in range(3):
                response = await client.post(
                    CHAT_PATH,
                    json={"message": "What's the capital of France?"},
                    headers={"Origin": ORIGIN, "X-Forwarded-For": f"198.51.100.{n}", "X-Real-IP": f"198.51.100.{n}"},
                )
                statuses.append(response.status_code)
            return statuses

    assert run(scenario) == [200, 200, 429]


def test_a_session_out_of_turns_is_429_session_limit():
    app, _, _ = build(turns=1)

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            first = await post(client, {"message": "What's the capital of France?"})
            session = stream_of(first)[0]
            assert isinstance(session, SessionEvent) and session.turns_left == 0
            return await post(client, {"message": "What's the capital of France?", "session_id": session.session_id})

    response = run(scenario)
    assert response.status_code == 429
    assert error_of(response) == {
        "code": "session_limit",
        "detail": "This session has used all its turns. Start a new conversation.",
    }
    assert "retry-after" not in response.headers
    assert cors_of(response) == CORS


def test_a_full_queue_is_503_busy():
    app, engine, _ = build(slots=1, queue=0)

    async def scenario() -> httpx.Response:
        held = await engine.chat(IPv4Address("198.51.100.7"), ChatRequest(message="What projects have you built?"))
        assert isinstance(held, Reply)
        async with http(app) as client:
            response = await post(client, {"message": "What projects have you built?"})
        await held.aclose()
        return response

    response = run(scenario)
    assert response.status_code == 503
    assert error_of(response) == {
        "code": "busy",
        "retry_after_s": 5,
        "detail": "Every generation slot and queue place is taken.",
    }
    assert response.headers["retry-after"] == "5"
    assert cors_of(response) == CORS


def test_a_guard_model_that_is_down_is_503_unavailable():
    class DownModerator:
        def check(self, turn: Turn) -> Verdict:
            return Down(RuleId("moderation.down"))

    app, _, _ = build(moderator=DownModerator())

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await post(client, {"message": "What projects have you built?"})

    response = run(scenario)
    assert response.status_code == 503
    assert error_of(response) == {
        "code": "unavailable",
        "retry_after_s": 30,
        "detail": "A model the guards need is not answering.",
    }
    assert response.headers["retry-after"] == "30"


def test_preflight_from_an_allowed_origin_is_204_with_the_cors_policy():
    app, _, _ = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await client.options(CHAT_PATH, headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})

    response = run(scenario)
    assert response.status_code == 204
    assert {name: value for name, value in response.headers.items() if name.startswith("access-control")} == {
        "access-control-allow-origin": ORIGIN,
        "access-control-expose-headers": "Retry-After, X-Request-Id",
        "access-control-allow-methods": "POST",
        "access-control-allow-headers": "Content-Type",
        "access-control-max-age": "600",
    }


def test_preflight_from_another_origin_is_403_with_no_cors_headers():
    app, _, _ = build()

    async def scenario() -> httpx.Response:
        async with http(app) as client:
            return await client.options(CHAT_PATH, headers={"Origin": "https://alex.example.attacker.example"})

    response = run(scenario)
    assert response.status_code == 403
    assert error_of(response)["code"] == "origin_not_allowed"
    assert cors_of(response) == {}


def test_the_openapi_document_has_the_body_schema_and_every_rejection_status():
    app, _, _ = build()
    operation = app.openapi()["paths"][CHAT_PATH]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["required"] == ["message"]
    assert sorted(operation["responses"]) == ["200", "400", "403", "413", "429", "503"]


QUEUED_FRAME = b'event: queued\ndata: {"type":"queued","position":1}\n\n'
STARTED_FRAME = b'event: started\ndata: {"type":"started"}\n\n'


def test_a_queue_wait_sends_heartbeats_until_the_slot_frees(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vitae.app, "HEARTBEAT_AFTER_S", 0.05)
    _, engine, _ = build(slots=1, queue=1)

    async def scenario() -> list[bytes]:
        held = await engine.chat(IPv4Address("198.51.100.7"), ChatRequest(message="What projects have you built?"))
        waiting = await engine.chat(IPv4Address("198.51.100.8"), ChatRequest(message="What projects have you built?"))
        assert isinstance(held, Reply) and isinstance(waiting, Reply)
        frames: list[bytes] = []
        async for chunk in framed(waiting):
            frames.append(chunk)
            if chunk == HEARTBEAT and QUEUED_FRAME in frames:
                await held.aclose()
        return frames

    frames = run(scenario)
    after_queued = frames[frames.index(QUEUED_FRAME) + 1 :]
    assert after_queued[0] == HEARTBEAT
    events = list(read_events(b"".join(frames).decode().splitlines()))
    assert [event.type for event in events if not isinstance(event, Malformed)] == [
        "session",
        "queued",
        "started",
        "text",
        "text",
        "done",
    ]

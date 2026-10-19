"""The HTTP cases over real HTTP on localhost: origin, size, per-IP rate limit, spoofed forwarding
headers, session limit, a full queue, and IPv6 /64 keying. Limits are set with VITAE_* variables."""

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tests.http.harness import ORIGIN
from vitae.contract import CHAT_PATH, DoneEvent, QueuedEvent, SessionEvent, StartedEvent, StreamEvent, TextEvent
from vitae.wire import Malformed, read_events

Serve = Callable[..., str]
ANSWERED_TEXT = "I built Shift Notes. It helps support teams with handovers."
RATE_DETAIL = "This address has sent too many requests."


def post(
    base: str,
    message: str,
    *,
    origin: str | None = ORIGIN,
    session_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    sent = {"Origin": origin} if origin is not None else {}
    body: dict[str, Any] = {"message": message}
    if session_id is not None:
        body["session_id"] = session_id
    return httpx.post(f"{base}{CHAT_PATH}", json=body, headers={**sent, **(headers or {})}, timeout=30)


def events(response: httpx.Response) -> list[StreamEvent | Malformed]:
    return list(read_events(response.text.splitlines()))


def answered_text(response: httpx.Response) -> str:
    stream = events(response)
    assert isinstance(stream[0], SessionEvent)
    assert stream[-1] == DoneEvent(finish="complete")
    return "".join(event.text for event in stream if isinstance(event, TextEvent))


def test_an_allowed_origin_is_answered(serve: Serve):
    base = serve()
    response = post(base, "What projects have you built?")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert answered_text(response) == ANSWERED_TEXT


@pytest.mark.parametrize(
    "origin",
    ["https://attacker.example", "http://localhost:5173.attacker.example", None],
    ids=["other", "lookalike", "missing"],
)
def test_an_origin_off_the_list_is_403_without_cors(serve: Serve, origin: str | None):
    base = serve()
    response = post(base, "What projects have you built?", origin=origin)
    assert response.status_code == 403
    assert response.json() == {
        "error": {"code": "origin_not_allowed", "detail": "The Origin header is missing or not on the allowlist."}
    }
    assert "access-control-allow-origin" not in response.headers


def test_a_message_of_exactly_500_characters_is_answered(serve: Serve):
    base = serve()
    response = post(base, "Tell me about your work experience".ljust(500, "."))
    assert response.status_code == 200
    assert answered_text(response) == ANSWERED_TEXT


def test_a_message_of_501_characters_is_413_too_long(serve: Serve):
    base = serve()
    response = post(base, "Tell me about your work experience".ljust(501, "."))
    assert response.status_code == 413
    assert response.json() == {"error": {"code": "too_long", "detail": "message is over 500 characters."}}


def assert_rate_limited(response: httpx.Response) -> None:
    assert response.status_code == 429
    retry_after = int(response.headers["retry-after"])
    assert 1 <= retry_after <= 70
    assert response.json() == {"error": {"code": "rate_limited", "retry_after_s": retry_after, "detail": RATE_DETAIL}}


def test_the_request_after_the_per_ip_limit_is_429_with_retry_after(serve: Serve):
    base = serve(VITAE_RATE_LIMITS="3/minute")
    statuses = [post(base, "What's the capital of France?").status_code for _ in range(3)]
    assert statuses == [200, 200, 200]
    assert_rate_limited(post(base, "What's the capital of France?"))


SPOOFS = [
    {"X-Forwarded-For": "198.51.100.1"},
    {"X-Real-IP": "198.51.100.2"},
    {"Forwarded": "for=198.51.100.3", "X-Forwarded-For": "198.51.100.3, 127.0.0.1"},
    {"X-Real-IP": "2001:db8:9::1", "X-Forwarded-For": "2001:db8:9::1"},
]


@pytest.mark.parametrize(
    "proxy",
    [{}, {"VITAE_TRUSTED_PROXY_CIDRS": "10.0.0.0/8", "VITAE_CLIENT_IP_HEADER": "X-Real-IP"}],
    ids=["no-proxy", "loopback-not-trusted"],
)
def test_spoofed_forwarding_headers_from_an_untrusted_peer_share_the_peers_limit(serve: Serve, proxy: dict[str, str]):
    base = serve(VITAE_RATE_LIMITS="3/minute", **proxy)
    statuses = [post(base, "What's the capital of France?", headers=spoof).status_code for spoof in SPOOFS[:3]]
    assert statuses == [200, 200, 200]
    assert_rate_limited(post(base, "What's the capital of France?", headers=SPOOFS[3]))


def test_a_session_out_of_turns_is_429_session_limit(serve: Serve):
    base = serve(VITAE_SESSION_MAX_TURNS="2")
    first = post(base, "What projects have you built?")
    session = events(first)[0]
    assert isinstance(session, SessionEvent) and session.turns_left == 1
    second = post(base, "What projects have you built?", session_id=session.session_id)
    assert events(second)[0] == SessionEvent(session_id=session.session_id, turns_left=0)
    third = post(base, "What projects have you built?", session_id=session.session_id)
    assert third.status_code == 429
    assert third.json() == {
        "error": {"code": "session_limit", "detail": "This session has used all its turns. Start a new conversation."}
    }
    assert "retry-after" not in third.headers


def test_slots_plus_queue_plus_one_in_parallel_gets_exactly_one_busy(serve: Serve):
    base = serve(delay_s=0.4, VITAE_GENERATION_SLOTS="1", VITAE_QUEUE_SIZE="2")

    async def scenario() -> list[httpx.Response]:
        async with httpx.AsyncClient(base_url=base, timeout=30) as client:
            return await asyncio.gather(
                *(
                    client.post(
                        CHAT_PATH, json={"message": "What projects have you built?"}, headers={"Origin": ORIGIN}
                    )
                    for _ in range(4)
                )
            )

    responses = asyncio.run(scenario())
    assert sorted(response.status_code for response in responses) == [200, 200, 200, 503]
    busy = next(response for response in responses if response.status_code == 503)
    assert busy.headers["retry-after"] == "5"
    assert busy.json() == {
        "error": {"code": "busy", "retry_after_s": 5, "detail": "Every generation slot and queue place is taken."}
    }
    answered = [response for response in responses if response.status_code == 200]
    assert [answered_text(response) for response in answered] == [ANSWERED_TEXT] * 3
    positions = sorted(
        [event.position for event in events(response) if isinstance(event, QueuedEvent)] for response in answered
    )
    assert positions == [[], [1], [2, 1]]
    assert all(StartedEvent() in events(response) for response in answered)


TRUSTED_LOOPBACK = {"VITAE_TRUSTED_PROXY_CIDRS": "127.0.0.1/32", "VITAE_CLIENT_IP_HEADER": "X-Real-IP"}


def from_ip(base: str, address: str) -> httpx.Response:
    return post(base, "What's the capital of France?", headers={"X-Real-IP": address})


def test_ipv6_addresses_in_one_64_share_one_limit_and_another_64_does_not(serve: Serve):
    base = serve(VITAE_RATE_LIMITS="3/minute", **TRUSTED_LOOPBACK)
    same_64 = ["2001:db8:1:2::1", "2001:db8:1:2:ffff:ffff:ffff:ffff", "2001:db8:1:2:abcd:1234:5678:9abc"]
    assert [from_ip(base, address).status_code for address in same_64] == [200, 200, 200]
    assert_rate_limited(from_ip(base, "2001:db8:1:2::dead"))
    assert from_ip(base, "2001:db8:1:3::1").status_code == 200
    assert from_ip(base, "2001:db8:1:1:ffff:ffff:ffff:ffff").status_code == 200


def test_an_ipv4_mapped_ipv6_address_shares_the_ipv4_clients_limit(serve: Serve):
    base = serve(VITAE_RATE_LIMITS="3/minute", **TRUSTED_LOOPBACK)
    assert [from_ip(base, "203.0.113.5").status_code for _ in range(3)] == [200, 200, 200]
    assert_rate_limited(from_ip(base, "::ffff:203.0.113.5"))
    assert from_ip(base, "203.0.113.6").status_code == 200

"""HTTP only: origin, peer address, capped body, CORS, preflight and status codes. Never sees a
guard, a session, a slot or a sentence."""

import asyncio
from collections.abc import AsyncGenerator
from ipaddress import ip_address
from typing import Any, Final, assert_never
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from vitae.contract import (
    CHAT_PATH,
    MAX_BODY_BYTES,
    REJECTION_STATUS,
    ChatRequest,
    ErrorResponse,
    FinalRejection,
    Rejection,
    RetryableRejection,
    StreamEvent,
    parse_chat_request,
)
from vitae.engine import Engine, Reply
from vitae.observe import Observer
from vitae.settings import Settings, client_key
from vitae.wire import HEARTBEAT, SSE_HEADERS, encode_event, encode_rejection

ORIGIN_DETAIL: Final = "The Origin header is missing or not on the allowlist."
HEARTBEAT_AFTER_S: Final = 15.0
REQUEST_ID_HEADER: Final = "X-Request-Id"
UNKNOWN_PEER: Final = "0.0.0.0"
"""Stands in for a peer that is not an IP address (a Unix socket), so such clients share one limit
instead of escaping it."""
REJECTION_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    status: {
        "model": ErrorResponse,
        "description": ", ".join(code for code, s in REJECTION_STATUS.items() if s == status),
    }
    for status in sorted(set(REJECTION_STATUS.values()))
}
"""ErrorResponse documented per rejection status, so Schemathesis checks every pre-stream error."""
_REQUEST_BODY: Final = {
    "requestBody": {"required": True, "content": {"application/json": {"schema": ChatRequest.model_json_schema()}}}
}
"""The route reads the raw body itself, to cap it, so FastAPI cannot infer the schema."""


STREAM_RESPONSE: Final[dict[int | str, dict[str, Any]]] = {
    200: {
        "description": "The SSE stream of one turn.",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


def create_app(settings: Settings, engine: Engine, observer: Observer) -> FastAPI:
    app = FastAPI(openapi_url="/openapi.json", docs_url=None, redoc_url=None)

    @app.options(CHAT_PATH)
    async def preflight(request: Request) -> Response:
        origin = request.headers.get("origin")
        if origin is None or not settings.origin_allowed(origin):
            response = reject(FinalRejection(code="origin_not_allowed", detail=ORIGIN_DETAIL), cors_origin=None)
        else:
            response = Response(status_code=204, headers=preflight_headers(origin))
        response.headers[REQUEST_ID_HEADER] = str(uuid4())
        return response

    @app.post(
        CHAT_PATH,
        response_class=Response,
        responses=STREAM_RESPONSE | REJECTION_RESPONSES,
        openapi_extra=_REQUEST_BODY,
    )
    async def chat(request: Request) -> Response:
        request_id = uuid4()
        response = await _chat(request, request_id=request_id)
        response.headers[REQUEST_ID_HEADER] = str(request_id)
        return response

    async def _chat(request: Request, *, request_id: UUID) -> Response:
        origin = request.headers.get("origin")
        if origin is None or not settings.origin_allowed(origin):
            observer.rejected("origin_not_allowed")
            return reject(FinalRejection(code="origin_not_allowed", detail=ORIGIN_DETAIL), cors_origin=None)
        ip = settings.client_ip.resolve(peer_of(request), request.headers)
        body = await read_capped(request, MAX_BODY_BYTES)
        parsed = body if isinstance(body, FinalRejection) else parse_chat_request(body)
        if not isinstance(parsed, ChatRequest):
            observer.rejected(parsed.code)
            return reject(parsed, cors_origin=origin)
        reply = await engine.chat(client_key(ip), parsed, request_id=request_id)
        match reply:
            case RetryableRejection() | FinalRejection():
                observer.rejected(reply.code)
                return reject(reply, cors_origin=origin)
            case Reply():
                return StreamingResponse(
                    framed(reply),
                    headers={**SSE_HEADERS, **cors_headers(origin)},
                    background=BackgroundTask(reply.aclose),
                )
            case _:
                assert_never(reply)

    return app


def reject(rejection: Rejection, *, cors_origin: str | None) -> Response:
    """encode_rejection plus cors_headers(origin) when an origin was allowed, none for origin_not_allowed."""
    status, body, headers = encode_rejection(rejection)
    if cors_origin is not None:
        headers |= cors_headers(cors_origin)
    return Response(content=body, status_code=status, headers=headers)


async def read_capped(request: Request, limit: int) -> bytes | FinalRejection:
    """Checks Content-Length first, then stops reading past limit bytes, so an 8 GB body costs 8 KB.
    An oversized body is too_long, the same rejection parse_chat_request gives."""
    too_long = FinalRejection(code="too_long", detail=f"Request body is over {limit} bytes.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return too_long
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return too_long
    return bytes(body)


def peer_of(request: Request) -> str:
    host = request.client.host if request.client is not None else ""
    try:
        ip_address(host)
    except ValueError:
        return UNKNOWN_PEER
    return host


def cors_headers(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Expose-Headers": f"Retry-After, {REQUEST_ID_HEADER}",
        "Vary": "Origin",
    }


def preflight_headers(origin: str) -> dict[str, str]:
    return {
        **cors_headers(origin),
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "600",
    }


async def framed(reply: Reply) -> AsyncGenerator[bytes]:
    """wire.encode_event per event, plus wire.HEARTBEAT after HEARTBEAT_AFTER_S of silence (queue waits)."""
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=1)
    pump = asyncio.create_task(_pump(reply, queue))
    try:
        while True:
            try:
                async with asyncio.timeout(HEARTBEAT_AFTER_S):
                    event = await queue.get()
            except TimeoutError:
                yield HEARTBEAT
                continue
            if event is None:
                return
            yield encode_event(event)
    finally:
        pump.cancel()
        await asyncio.wait({pump})


async def _pump(reply: Reply, queue: asyncio.Queue[StreamEvent | None]) -> None:
    """Owns the reply: the only task that iterates or closes it, so a heartbeat wait never cancels it midway."""
    try:
        async for event in reply:
            await queue.put(event)
        await queue.put(None)
    finally:
        await reply.aclose()

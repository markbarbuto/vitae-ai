"""Bytes on the wire: SSE frames, error bodies, and a minimal SSE reader for Python clients."""

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from vitae.contract import STREAM_EVENT, ErrorResponse, Rejection, RetryableRejection, StreamEvent, rejection_status

SSE_MEDIA_TYPE: Final = "text/event-stream; charset=utf-8"
SSE_HEADERS: Final[Mapping[str, str]] = {
    "Content-Type": SSE_MEDIA_TYPE,
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",
}
HEARTBEAT: Final = b": keep-alive\n\n"


@dataclass(frozen=True)
class Malformed:
    """A data payload that is not a StreamEvent."""

    raw: str


def encode_event(event: StreamEvent) -> bytes:
    # ensure_ascii keeps U+2028, U+0085 and friends out of the frame: some line readers
    # (str.splitlines, httpx.iter_lines) split on them and would cut the data line in two.
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))
    return f"event: {event.type}\ndata: {data}\n\n".encode()


def encode_rejection(rejection: Rejection) -> tuple[int, bytes, dict[str, str]]:
    """(status, ErrorResponse JSON, headers). Retry-After mirrors retry_after_s."""
    body = ErrorResponse(error=rejection).model_dump_json().encode()
    headers = {"Content-Type": "application/json"}
    if isinstance(rejection, RetryableRejection):
        headers["Retry-After"] = str(rejection.retry_after_s)
    return rejection_status(rejection), body, headers


def read_events(lines: Iterable[str]) -> Iterator[StreamEvent | Malformed]:
    """Yields one item per complete frame. Comments and the event: line are ignored, because the
    payload's type is authoritative. A frame cut off before its blank line is dropped, as SSE specifies."""
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield _parse("\n".join(data))
                data = []
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").removeprefix(" "))


def _parse(raw: str) -> StreamEvent | Malformed:
    try:
        return STREAM_EVENT.validate_json(raw)
    except ValidationError:
        return Malformed(raw)

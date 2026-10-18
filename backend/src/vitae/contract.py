"""The HTTP and SSE contract for POST /v1/chat. Source of truth for contract/vitae-contract.*"""

from collections.abc import Mapping
from typing import Annotated, Final, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

CHAT_PATH: Final = "/v1/chat"
MAX_MESSAGE_CHARS: Final = 500
MAX_BODY_BYTES: Final = 8 * 1024


class Wire(BaseModel):
    """Base for every wire model. Unknown fields are a 400, never silently dropped."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_serialization_defaults_required=True,
        use_attribute_docstrings=True,
    )


SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{22}$")]
Text = Annotated[str, StringConstraints(min_length=1)]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
ActionHref = Annotated[str, StringConstraints(pattern=r"^(#|/|https?://|mailto:|tel:)\S*$")]
"""An in-page anchor, a same-site path, a web URL, mailto: or tel:. Never javascript: or data:, which run in the host page."""


class ChatRequest(Wire):
    """One visitor message. History lives server-side under session_id; the client never sends it."""

    message: Annotated[str, StringConstraints(max_length=MAX_MESSAGE_CHARS, pattern=r"\S")]
    """Not blank. maxLength counts code points, so a JS counter must use [...s].length, not s.length."""
    session_id: SessionId | None = None
    """Omit on the first turn. Unknown or expired ids start a new session."""


RetryableCode = Literal["rate_limited", "busy", "unavailable"]
FinalCode = Literal["invalid_request", "origin_not_allowed", "too_long", "session_limit"]
RejectionCode = Literal[RetryableCode, FinalCode]

REJECTION_STATUS: Final[Mapping[RejectionCode, int]] = {
    "invalid_request": 400,  # malformed JSON, unknown field, blank message, bad session_id shape
    "origin_not_allowed": 403,  # Origin missing or not an exact allowlist entry
    "too_long": 413,  # message over MAX_MESSAGE_CHARS or body over MAX_BODY_BYTES
    "rate_limited": 429,  # per-IP window exhausted
    "session_limit": 429,  # this session used its turns; start a new conversation
    "busy": 503,  # every generation slot and queue place taken
    "unavailable": 503,  # a model server or the index is down
}


class RetryableRejection(Wire):
    """The request may succeed after retry_after_s seconds."""

    code: RetryableCode
    retry_after_s: PositiveInt
    detail: str
    """Developer-facing English. The UI shows its own copy per code."""


class FinalRejection(Wire):
    """Retrying the same request cannot succeed."""

    code: FinalCode
    detail: str
    """Developer-facing English. The UI shows its own copy per code."""


Rejection = RetryableRejection | FinalRejection


class ErrorResponse(Wire):
    """Body of every non-200 response from POST /v1/chat."""

    error: Annotated[Rejection, Field(discriminator="code")]


def rejection_status(rejection: Rejection) -> int:
    return REJECTION_STATUS[rejection.code]


def parse_chat_request(body: bytes) -> ChatRequest | Rejection:
    """Boundary parser. An oversized body or message is too_long; anything else wrong is invalid_request."""
    if len(body) > MAX_BODY_BYTES:
        return FinalRejection(code="too_long", detail=f"Request body is over {MAX_BODY_BYTES} bytes.")
    try:
        return ChatRequest.model_validate_json(body)
    except ValidationError as error:
        problems = error.errors(include_url=False)
        if any(p["type"] == "string_too_long" and p["loc"] == ("message",) for p in problems):
            return FinalRejection(code="too_long", detail=f"message is over {MAX_MESSAGE_CHARS} characters.")
        first = problems[0]
        where = ".".join(str(part) for part in first["loc"]) or "body"
        return FinalRejection(code="invalid_request", detail=f"{where}: {first['msg']}")


RefusalReason = Literal[
    "unrelated", "blocked_topic", "injection", "pii", "unsafe", "insufficient_evidence", "output_rule"
]
StreamErrorCode = Literal["generation_failed", "timeout", "internal"]
Finish = Literal["complete", "length", "withheld"]


class Source(Wire):
    """A knowledge document the answer drew on."""

    id: Slug
    title: Text
    href: HttpUrl | None = None


class HostAction(Wire):
    """Owner-defined action from the profile. The UI calls onAction(id) if the host passed one, else follows href."""

    id: Slug
    label: Text
    href: ActionHref | None = None


class SessionEvent(Wire):
    """Always first. The client keeps session_id in memory (no cookie, no storage) and sends it next turn."""

    type: Literal["session"] = "session"
    session_id: SessionId
    turns_left: Annotated[int, Field(ge=0)]
    """0 means the next request gets 429 session_limit."""


class QueuedEvent(Wire):
    """Sent when the request waits for a generation slot, and again whenever its position changes."""

    type: Literal["queued"] = "queued"
    position: PositiveInt
    """1 means next to run."""


class StartedEvent(Wire):
    """A generation slot was acquired. The UI switches from queue position to the typing indicator."""

    type: Literal["started"] = "started"


class TextEvent(Wire):
    """One sentence that passed the output rules. Append verbatim; the server includes separating spaces."""

    type: Literal["text"] = "text"
    text: Text


class SourcesEvent(Wire):
    """Optional, at most once, after the last text and before done."""

    type: Literal["sources"] = "sources"
    sources: Annotated[tuple[Source, ...], Field(min_length=1)]


class DoneEvent(Wire):
    """Terminal. complete: model stopped. length: token cap hit. withheld: a later sentence failed the output rules."""

    type: Literal["done"] = "done"
    finish: Finish


class RefusedEvent(Wire):
    """Terminal. Only legal before any text event. message is the profile's refusal copy for this reason."""

    type: Literal["refused"] = "refused"
    reason: RefusalReason
    message: Text


class AvailabilityEvent(Wire):
    """Terminal. The profile's fixed availability answer, never generated."""

    type: Literal["availability"] = "availability"
    message: Text
    action: HostAction | None = None


class ErrorEvent(Wire):
    """Terminal. Something broke after the 200 was sent. Earlier text events stay on screen."""

    type: Literal["error"] = "error"
    code: StreamErrorCode
    detail: str


StreamEvent = (
    SessionEvent
    | QueuedEvent
    | StartedEvent
    | TextEvent
    | SourcesEvent
    | DoneEvent
    | RefusedEvent
    | AvailabilityEvent
    | ErrorEvent
)
TerminalEvent = DoneEvent | RefusedEvent | AvailabilityEvent | ErrorEvent
STREAM_EVENT: Final = TypeAdapter[StreamEvent](Annotated[StreamEvent, Field(discriminator="type")])


OUTCOMES: Final[tuple[str, ...]] = (
    "answered",
    "availability",
    "broken",
    *(f"refused:{reason}" for reason in get_args(RefusalReason)),
    *(f"rejected:{code}" for code in get_args(RejectionCode)),
)

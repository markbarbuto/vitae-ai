"""Reference semantics of one turn: fold a stream into what the visitor saw and how the evals grade it."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import ValidationError

from vitae.contract import (
    AvailabilityEvent,
    DoneEvent,
    ErrorEvent,
    ErrorResponse,
    QueuedEvent,
    RefusedEvent,
    SessionEvent,
    Source,
    SourcesEvent,
    StartedEvent,
    StreamEvent,
    TerminalEvent,
    TextEvent,
    rejection_status,
)
from vitae.wire import Malformed, read_events

Outcome = str
"""One of contract.OUTCOMES."""

Phase = Literal["start", "waiting", "running", "answering", "sourced", "ended"]

# stream = session queued* started? ( early | answer );
# early = refused | availability | error; answer = text+ sources? ( done | error )
GRAMMAR: Final[Mapping[Phase, Mapping[str, Phase]]] = {
    "start": {"session": "waiting"},
    "waiting": {
        "queued": "waiting",
        "started": "running",
        "text": "answering",
        "refused": "ended",
        "availability": "ended",
        "error": "ended",
    },
    "running": {"text": "answering", "refused": "ended", "availability": "ended", "error": "ended"},
    "answering": {"text": "answering", "sources": "sourced", "done": "ended", "error": "ended"},
    "sourced": {"done": "ended", "error": "ended"},
    "ended": {},
}


@dataclass(frozen=True)
class Transcript:
    """What one visitor saw for one request, and how the evals grade it."""

    outcome: Outcome
    text: str
    """Released sentences, then the refusal or availability message."""
    streamed_then_failed: bool
    """Text was shown and the turn then ended in refused, error or broken."""
    session_id: str | None
    sources: tuple[Source, ...]
    first_text_ms: int | None
    """Time from request to the first text event."""


def fold(events: Iterable[tuple[float, StreamEvent | Malformed]], started: float) -> Transcript:
    """Pure. A grammar violation, a Malformed event, or a stream that ends without a terminal is
    "broken". So is error, with streamed_then_failed set when text was already shown."""
    phase: Phase = "start"
    terminal: TerminalEvent | None = None
    texts: list[str] = []
    session_id: str | None = None
    sources: tuple[Source, ...] = ()
    first_text_ms: int | None = None

    for at, event in events:
        next_phase = None if isinstance(event, Malformed) else GRAMMAR[phase].get(event.type)
        if next_phase is None:
            terminal = None
            break
        phase = next_phase
        match event:
            case SessionEvent():
                session_id = event.session_id
            case TextEvent():
                if first_text_ms is None:
                    first_text_ms = round((at - started) * 1000)
                texts.append(event.text)
            case SourcesEvent():
                sources = event.sources
            case RefusedEvent() | AvailabilityEvent():
                texts.append(event.message)
                terminal = event
            case DoneEvent() | ErrorEvent():
                terminal = event
            case QueuedEvent() | StartedEvent() | Malformed():
                pass

    outcome = _outcome(terminal) if phase == "ended" and terminal is not None else "broken"
    return Transcript(
        outcome=outcome,
        text="".join(texts),
        streamed_then_failed=first_text_ms is not None and outcome == "broken",
        session_id=session_id,
        sources=sources,
        first_text_ms=first_text_ms,
    )


def _outcome(terminal: TerminalEvent) -> Outcome:
    match terminal:
        case DoneEvent():
            return "answered"
        case AvailabilityEvent():
            return "availability"
        case RefusedEvent():
            return f"refused:{terminal.reason}"
        case ErrorEvent():
            return "broken"


def transcript_of_response(status: int, lines: Iterable[str], clock: Callable[[], float], started: float) -> Transcript:
    """A non-200 status is a rejection read from the ErrorResponse body. A 200 is a stream, each
    event stamped with clock() as it arrives."""
    if status != 200:
        return _rejected(status, "\n".join(lines))
    return fold(((clock(), event) for event in read_events(lines)), started)


def _rejected(status: int, body: str) -> Transcript:
    try:
        rejection = ErrorResponse.model_validate_json(body).error
        outcome = f"rejected:{rejection.code}" if rejection_status(rejection) == status else "broken"
    except ValidationError:
        outcome = "broken"
    return Transcript(
        outcome=outcome, text="", streamed_then_failed=False, session_id=None, sources=(), first_text_ms=None
    )

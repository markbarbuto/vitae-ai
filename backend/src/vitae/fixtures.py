"""Example turns, written to contract/streams/ for the Python tests now and the TS reducer tests in phase 5.

Each fixture is what the server sends plus what a visitor must end up seeing. Some are deliberately
broken, so every client proves it rejects them the same way transcript.fold does."""

from dataclasses import dataclass
from typing import Final

from vitae.contract import (
    AvailabilityEvent,
    DoneEvent,
    ErrorEvent,
    FinalRejection,
    HostAction,
    QueuedEvent,
    RefusedEvent,
    Rejection,
    RetryableRejection,
    SessionEvent,
    Source,
    SourcesEvent,
    StartedEvent,
    StreamEvent,
    TextEvent,
)
from vitae.transcript import Outcome, Transcript
from vitae.wire import HEARTBEAT

SESSION: Final = "c2Vzc2lvbi1leGFtcGxlMQ"


@dataclass(frozen=True)
class Raw:
    """Bytes written verbatim between event frames: a heartbeat, or a frame no valid server sends."""

    frame: bytes


@dataclass(frozen=True)
class Expected:
    """The Transcript fields a fixture pins. Timing is left out because it depends on the reader's clock."""

    outcome: Outcome
    text: str
    streamed_then_failed: bool
    session_id: str | None
    sources: tuple[Source, ...]

    @classmethod
    def of(cls, transcript: Transcript) -> "Expected":
        return cls(
            outcome=transcript.outcome,
            text=transcript.text,
            streamed_then_failed=transcript.streamed_then_failed,
            session_id=transcript.session_id,
            sources=transcript.sources,
        )


@dataclass(frozen=True)
class StreamFixture:
    name: str
    frames: tuple[StreamEvent | Raw, ...]
    expected: Expected


@dataclass(frozen=True)
class RejectionFixture:
    name: str
    rejection: Rejection
    expected: Expected


Fixture = StreamFixture | RejectionFixture

_PROJECTS = Source(id="projects", title="Projects")

FIXTURES: Final[tuple[Fixture, ...]] = (
    StreamFixture(
        name="answered",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            QueuedEvent(position=2),
            QueuedEvent(position=1),
            Raw(HEARTBEAT),
            StartedEvent(),
            TextEvent(text="I built Shift Notes, a handoff tool for support teams. "),
            TextEvent(text="I also built Neighbourhood Signals, which maps local service requests."),
            SourcesEvent(sources=(_PROJECTS,)),
            DoneEvent(finish="complete"),
        ),
        expected=Expected(
            outcome="answered",
            text="I built Shift Notes, a handoff tool for support teams. "
            "I also built Neighbourhood Signals, which maps local service requests.",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(_PROJECTS,),
        ),
    ),
    StreamFixture(
        name="withheld",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=18),
            StartedEvent(),
            TextEvent(text="My main projects are Shift Notes and Neighbourhood Signals. "),
            DoneEvent(finish="withheld"),
        ),
        expected=Expected(
            outcome="answered",
            text="My main projects are Shift Notes and Neighbourhood Signals. ",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="refused-pii",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            RefusedEvent(
                reason="pii",
                message="Please don't share personal details like email addresses or phone numbers here.",
            ),
        ),
        expected=Expected(
            outcome="refused:pii",
            text="Please don't share personal details like email addresses or phone numbers here.",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="availability",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            StartedEvent(),
            AvailabilityEvent(
                message="I'm open to new roles. The best way to reach me is through the contact form.",
                action=HostAction(id="contact", label="Get in touch", href="#contact"),
            ),
        ),
        expected=Expected(
            outcome="availability",
            text="I'm open to new roles. The best way to reach me is through the contact form.",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="error-before-text",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            StartedEvent(),
            ErrorEvent(code="timeout", detail="The model server did not answer in time."),
        ),
        expected=Expected(
            outcome="broken",
            text="",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="error-after-text",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            StartedEvent(),
            TextEvent(text="I built Shift Notes, a handoff tool for support teams. "),
            ErrorEvent(code="generation_failed", detail="The model server closed the connection."),
        ),
        expected=Expected(
            outcome="broken",
            text="I built Shift Notes, a handoff tool for support teams. ",
            streamed_then_failed=True,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="broken-refused-after-text",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            TextEvent(text="Sure, here is the link: "),
            RefusedEvent(reason="output_rule", message="I can't share that."),
        ),
        expected=Expected(
            outcome="broken",
            text="Sure, here is the link: ",
            streamed_then_failed=True,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="broken-queued-after-started",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            StartedEvent(),
            QueuedEvent(position=1),
            TextEvent(text="I built Shift Notes, a handoff tool for support teams. "),
            DoneEvent(finish="complete"),
        ),
        expected=Expected(
            outcome="broken",
            text="",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="broken-malformed",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            StartedEvent(),
            Raw(b'event: text\ndata: {"type":"text","text":\n\n'),
        ),
        expected=Expected(
            outcome="broken",
            text="",
            streamed_then_failed=False,
            session_id=SESSION,
            sources=(),
        ),
    ),
    StreamFixture(
        name="broken-no-terminal",
        frames=(
            SessionEvent(session_id=SESSION, turns_left=19),
            StartedEvent(),
            TextEvent(text="I built Shift Notes, a handoff tool for support teams. "),
        ),
        expected=Expected(
            outcome="broken",
            text="I built Shift Notes, a handoff tool for support teams. ",
            streamed_then_failed=True,
            session_id=SESSION,
            sources=(),
        ),
    ),
    RejectionFixture(
        name="rejected-rate-limited",
        rejection=RetryableRejection(code="rate_limited", retry_after_s=42, detail="6 requests per minute per IP"),
        expected=Expected(
            outcome="rejected:rate_limited",
            text="",
            streamed_then_failed=False,
            session_id=None,
            sources=(),
        ),
    ),
    RejectionFixture(
        name="rejected-origin",
        rejection=FinalRejection(code="origin_not_allowed", detail="Origin is not on this server's allowlist."),
        expected=Expected(
            outcome="rejected:origin_not_allowed",
            text="",
            streamed_then_failed=False,
            session_id=None,
            sources=(),
        ),
    ),
)

"""One turn, start to finish: throttle, screen, admit, queue, prompt, generate, release, remember, record."""

import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass, field
from time import perf_counter
from typing import Final, Literal, Self
from uuid import UUID, uuid4

from vitae.answer import OutputRules, Releaser, availability_event, refusal_event
from vitae.contract import (
    ChatRequest,
    ErrorEvent,
    QueuedEvent,
    Rejection,
    RetryableRejection,
    SessionEvent,
    StartedEvent,
    StreamEvent,
    TerminalEvent,
)
from vitae.gate import GUARD_WORKERS, Admitted, Exchange, Gate, GateLimits, Generate, ScreenPermit, Stop
from vitae.generation import FAILED, Delta, Generator
from vitae.guards import Availability, InputStages, Refuse, RuleId, StageName, build_input_stages
from vitae.guards.injection import Calibration, InjectionClassifier
from vitae.guards.moderation import NO_MODERATION, Moderator
from vitae.observe import NULL_OBSERVER, Observer, TurnRecord
from vitae.profile import Profile
from vitae.prompt import build_messages, system_prompt
from vitae.screen import NO_EVIDENCE, Proceed, Retriever, Screening, Unavailable, screen
from vitae.settings import ClientKey, Settings
from vitae.text import Turn, normalize
from vitae.transcript import fold

UNAVAILABLE_RETRY_S: Final = 30
UNAVAILABLE_DETAIL: Final = "A model the guards need is not answering."
INTERNAL_DETAIL: Final = "The server hit an unexpected error."


class Reply:
    """The stream of one admitted turn. Async-iterable once. Releases its lease on exhaustion,
    aclose(), or garbage collection, whichever comes first."""

    def __init__(self, events: AsyncGenerator[StreamEvent], admitted: Admitted) -> None:
        self._events = events
        # A disconnect before the first event means the generator never starts, so its finally never
        # runs. finalize releases on drop, and calling it from aclose releases at most once.
        self._release = weakref.finalize(self, admitted.lease.release) if isinstance(admitted, Generate) else None

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._events

    async def aclose(self) -> None:
        await self._events.aclose()
        if self._release is not None:
            self._release()


@dataclass(slots=True)
class _Trace:
    """What one turn has done so far, for its TurnRecord. Per request, never shared."""

    screening: Screening
    decided_by: StageName | Literal["evidence", "output"] | None = None
    rule: RuleId | None = None
    stage_ms: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    queue_wait_ms: float | None = None
    first_delta_ms: float | None = None
    first_release_ms: float | None = None
    sentences_released: int = 0
    sentences_withheld: int = 0

    def record(self, request_id: UUID, sent: list[StreamEvent]) -> TurnRecord:
        return TurnRecord(
            request_id=request_id,
            outcome=fold(((0.0, event) for event in sent), 0.0).outcome,
            decided_by=self.decided_by,
            rule=self.rule,
            stage_ms=(*self.screening.timings, *self.stage_ms),
            queue_wait_ms=self.queue_wait_ms,
            first_delta_ms=self.first_delta_ms,
            first_release_ms=self.first_release_ms,
            sentences_released=self.sentences_released,
            sentences_withheld=self.sentences_withheld,
        )


def _ms_since(started: float) -> float:
    return (perf_counter() - started) * 1000.0


@dataclass(frozen=True, slots=True)
class Engine:
    gate: Gate
    stages: InputStages
    retrieve: Retriever
    profile: Profile
    generator: Generator
    output_rules: OutputRules
    observer: Observer
    clock: Callable[[], float] = time.monotonic
    guard_pool: Executor = field(default_factory=lambda: ThreadPoolExecutor(GUARD_WORKERS))

    @classmethod
    def build(
        cls,
        settings: Settings,
        profile: Profile,
        *,
        generator: Generator,
        classifier: InjectionClassifier,
        moderator: Moderator = NO_MODERATION,
        retrieve: Retriever = lambda _: NO_EVIDENCE,
        observer: Observer = NULL_OBSERVER,
    ) -> Self:
        return cls(
            gate=Gate(GateLimits.of(settings)),
            stages=build_input_stages(
                profile, classifier=classifier, moderator=moderator, calibration=Calibration.load()
            ),
            retrieve=retrieve,
            profile=profile,
            generator=generator,
            output_rules=OutputRules.build(
                system_prompt(profile),
                allowed_links=profile.output.allowed_links,
                allowed_contacts=profile.output.allowed_contacts,
                allowed_names=profile.output.allowed_names,
                refusal=profile.refusals["output_rule"],
            ),
            observer=observer,
        )

    async def chat(self, key: ClientKey, request: ChatRequest, *, request_id: UUID | None = None) -> Rejection | Reply:
        """Every pre-stream decision happens here, before the 200. Returns a Rejection the route
        turns into a status code, or a Reply whose events are the whole stream."""
        permit = self.gate.throttle(key, request.session_id, self.clock())
        if not isinstance(permit, ScreenPermit):
            return permit
        loop = asyncio.get_running_loop()
        try:
            turn = normalize(request.message)
            work = self.guard_pool.submit(screen, turn, self.stages, self.retrieve)
        except BaseException:
            permit.release()
            raise
        # The permit bounds classifier work, so it is freed when the work ends. A cancelled request
        # stops waiting while a started screen keeps running in its thread.
        work.add_done_callback(lambda _: loop.call_soon_threadsafe(permit.release))
        try:
            screening = await asyncio.wrap_future(work)
        finally:
            if work.done():
                permit.release()
        if isinstance(screening.decision, Unavailable):
            return RetryableRejection(code="unavailable", retry_after_s=UNAVAILABLE_RETRY_S, detail=UNAVAILABLE_DETAIL)
        admitted = self.gate.admit(
            request.session_id, generate=isinstance(screening.decision, Proceed), now=self.clock()
        )
        match admitted:
            case Generate() | Stop():
                return Reply(self._events(admitted, turn, screening, request_id or uuid4()), admitted)
            case _:
                return admitted

    async def _events(
        self, admitted: Admitted, turn: Turn, screening: Screening, request_id: UUID
    ) -> AsyncGenerator[StreamEvent]:
        trace = _Trace(screening)
        sent: list[StreamEvent] = []
        try:
            async with aclosing(self._turn(admitted, turn, screening, trace)) as events:
                async for event in events:
                    sent.append(event)
                    yield event
        except Exception:  # noqa: BLE001 - the one catch: a bug becomes a contract event, not a torn connection
            if not (sent and isinstance(sent[-1], TerminalEvent)):
                sent.append(ErrorEvent(code="internal", detail=INTERNAL_DETAIL))
                yield sent[-1]
        finally:
            if isinstance(admitted, Generate):
                admitted.lease.release()
            self.observer.turn(trace.record(request_id, sent))

    async def _turn(
        self, admitted: Admitted, turn: Turn, screening: Screening, trace: _Trace
    ) -> AsyncGenerator[StreamEvent]:
        yield SessionEvent(session_id=admitted.session.id, turns_left=admitted.session.turns_left)
        match admitted, screening.decision:
            case Stop(), Refuse(reason=reason, rule=rule):
                trace.decided_by, trace.rule = screening.decided_by, rule
                yield refusal_event(self.profile, reason)
            case Stop(), Availability(rule=rule):
                trace.decided_by, trace.rule = screening.decided_by, rule
                yield availability_event(self.profile)
            case Generate(lease=lease, session=session), Proceed(evidence=evidence):
                started = perf_counter()
                async for position in lease.positions():
                    yield QueuedEvent(position=position)
                trace.queue_wait_ms = _ms_since(started)
                trace.stage_ms.append(("queue", trace.queue_wait_ms))
                yield StartedEvent()

                started = perf_counter()
                messages = build_messages(self.profile, session.history, turn, evidence)
                trace.stage_ms.append(("prompt", _ms_since(started)))

                releaser = Releaser(self.output_rules, evidence.sources)
                release_ms = 0.0
                generation_started = perf_counter()
                async with aclosing(self.generator.stream(messages)) as stream:
                    async for event in stream:
                        if trace.first_delta_ms is None and isinstance(event, Delta):
                            trace.first_delta_ms = _ms_since(generation_started)
                        started = perf_counter()
                        out = releaser.feed(event)
                        release_ms += _ms_since(started)
                        if trace.first_release_ms is None and releaser.sentences_released:
                            trace.first_release_ms = _ms_since(generation_started)
                        for released in out:
                            yield released
                        if releaser.finished:
                            break
                if not releaser.finished:
                    for released in releaser.feed(FAILED):
                        yield released
                lease.release()
                trace.stage_ms.append(("release", release_ms))
                trace.sentences_released = releaser.sentences_released
                if releaser.violation is not None:
                    trace.decided_by, trace.rule = "output", releaser.violation.rule
                    trace.sentences_withheld = 1
                if releaser.released_text:
                    self.gate.remember(session.id, Exchange(turn.text, releaser.released_text), self.clock())
            case _:
                raise AssertionError(f"admitted {admitted!r} does not fit {screening.decision!r}")

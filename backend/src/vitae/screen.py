"""Runs the input stages in order, times them, then asks the evidence slot."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Final, Literal, assert_never

from vitae.contract import Source
from vitae.guards import Availability, Down, InputStages, Pass, Refuse, RuleId, Skipped, StageName
from vitae.text import Turn


@dataclass(frozen=True, slots=True)
class Evidence:
    passages: tuple[str, ...]
    """Phase 3 fills these."""
    sources: tuple[Source, ...]


NO_EVIDENCE: Final = Evidence((), ())


@dataclass(frozen=True, slots=True)
class Proceed:
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class Unavailable:
    rule: RuleId
    """From a Down verdict or a down index. The route answers 503 unavailable."""


Decision = Refuse | Availability | Proceed | Unavailable
"""What screening concludes. Pass and Skipped are not decisions: they only mean "ask the next stage".
Unavailable is decided before the 200, so the route answers it with a real 503."""

Retriever = Callable[[Turn], Evidence | Refuse]
"""Phase 3 slot. Returns Refuse("insufficient_evidence", ...) when the evidence score is too low.
Phase 2 wires `lambda _: NO_EVIDENCE`."""


@dataclass(frozen=True, slots=True)
class Screening:
    decision: Decision
    decided_by: StageName | Literal["evidence"] | None
    timings: tuple[tuple[str, float], ...]
    """(stage, ms) in run order, for the stages that ran."""


def screen(turn: Turn, stages: InputStages, retrieve: Retriever = lambda _: NO_EVIDENCE) -> Screening:
    """Sync and pure apart from perf_counter. First non-Pass verdict wins; later stages do not run.
    The engine calls this in its bounded guard executor; the benchmark calls it directly."""
    timings: list[tuple[str, float]] = []
    for stage in stages.stages:
        started = perf_counter()
        verdict = stage.check(turn)
        timings.append((stage.name, _ms_since(started)))
        match verdict:
            case Pass() | Skipped():
                continue
            case Refuse() | Availability():
                return Screening(verdict, stage.name, tuple(timings))
            case Down(rule=rule):
                return Screening(Unavailable(rule), stage.name, tuple(timings))
            case _:
                assert_never(verdict)
    started = perf_counter()
    found = retrieve(turn)
    timings.append(("evidence", _ms_since(started)))
    match found:
        case Refuse():
            return Screening(found, "evidence", tuple(timings))
        case Evidence():
            return Screening(Proceed(found), None, tuple(timings))


def _ms_since(started: float) -> float:
    return (perf_counter() - started) * 1000.0

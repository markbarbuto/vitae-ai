"""What a turn leaves behind in the logs. The record type has nowhere to put visitor text or an address."""

import json
from dataclasses import asdict, dataclass
from typing import Final, Literal, Protocol, TextIO
from uuid import UUID

from vitae.contract import RejectionCode
from vitae.guards import RuleId, StageName


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """Everything one turn leaves in the logs. Deliberately has no field that can hold visitor text,
    model text or an address: only ids, enums and numbers."""

    request_id: UUID
    outcome: str
    """One of contract.OUTCOMES."""
    decided_by: StageName | Literal["evidence", "output"] | None
    rule: RuleId | None
    stage_ms: tuple[tuple[str, float], ...]
    """From Screening.timings, plus "queue", "prompt" and "release"."""
    queue_wait_ms: float | None
    first_delta_ms: float | None
    """First token from the generator."""
    first_release_ms: float | None
    """First TextEvent. The difference from first_delta_ms is the sentence-buffer cost (T6)."""
    sentences_released: int
    sentences_withheld: int


class Observer(Protocol):
    def turn(self, record: TurnRecord) -> None: ...
    def rejected(self, code: RejectionCode) -> None: ...


class JsonLogObserver:
    """One JSON line per call. Phase 4 adds a Prometheus observer behind the same protocol."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def turn(self, record: TurnRecord) -> None:
        fields = asdict(record)
        fields["request_id"] = str(record.request_id)
        fields["stage_ms"] = {stage: round(ms, 3) for stage, ms in record.stage_ms}
        self._write({"event": "turn", **fields})

    def rejected(self, code: RejectionCode) -> None:
        self._write({"event": "rejected", "code": code})

    def _write(self, line: dict[str, object]) -> None:
        self._stream.write(json.dumps(line, separators=(",", ":")) + "\n")
        self._stream.flush()


class _NullObserver:
    def turn(self, record: TurnRecord) -> None:
        pass

    def rejected(self, code: RejectionCode) -> None:
        pass


NULL_OBSERVER: Final[Observer] = _NullObserver()

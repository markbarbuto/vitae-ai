import io
import json
from dataclasses import fields
from uuid import UUID

from vitae.guards import RuleId
from vitae.observe import JsonLogObserver, TurnRecord


def test_a_turn_is_one_json_line_of_ids_enums_and_numbers():
    out = io.StringIO()
    observer = JsonLogObserver(out)
    observer.turn(
        TurnRecord(
            request_id=UUID("12345678-1234-4234-8234-123456789abc"),
            outcome="answered",
            decided_by="output",
            rule=RuleId("output.url"),
            stage_ms=(("pii", 0.12345), ("queue", 3.0)),
            queue_wait_ms=3.0,
            first_delta_ms=40.5,
            first_release_ms=52.25,
            sentences_released=1,
            sentences_withheld=1,
        )
    )
    observer.rejected("busy")
    first, second = out.getvalue().splitlines()
    assert json.loads(first) == {
        "event": "turn",
        "request_id": "12345678-1234-4234-8234-123456789abc",
        "outcome": "answered",
        "decided_by": "output",
        "rule": "output.url",
        "stage_ms": {"pii": 0.123, "queue": 3.0},
        "queue_wait_ms": 3.0,
        "first_delta_ms": 40.5,
        "first_release_ms": 52.25,
        "sentences_released": 1,
        "sentences_withheld": 1,
    }
    assert json.loads(second) == {"event": "rejected", "code": "busy"}


def test_the_turn_record_has_no_field_that_holds_free_text():
    assert {field.name: field.type for field in fields(TurnRecord) if field.type in (str, "str")} == {"outcome": str}

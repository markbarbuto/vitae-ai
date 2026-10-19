import json
from itertools import count
from pathlib import Path
from typing import get_args

import pytest

from vitae.contract import OUTCOMES, StreamEvent
from vitae.fixtures import FIXTURES, Expected, Fixture, StreamFixture
from vitae.transcript import Transcript, transcript_of_response

STREAMS = Path(__file__).parents[2] / "contract" / "streams"
EXPECTED = json.loads((STREAMS / "expected.json").read_text(encoding="utf-8"))


def ticking_clock(step_s: float = 0.1):
    return count(start=step_s, step=step_s).__next__


def replay(name: str) -> Transcript:
    entry = EXPECTED[name]
    lines = (STREAMS / entry["file"]).read_text(encoding="utf-8").splitlines()
    return transcript_of_response(entry["status"], lines, clock=ticking_clock(), started=0.0)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
def test_every_committed_example_folds_to_its_expected_transcript(fixture: Fixture):
    assert Expected.of(replay(fixture.name)) == fixture.expected


@pytest.mark.parametrize(
    ("name", "outcome", "streamed_then_failed"),
    [
        ("answered", "answered", False),
        ("withheld", "answered", False),
        ("refused-pii", "refused:pii", False),
        ("availability", "availability", False),
        ("error-before-text", "broken", False),
        ("error-after-text", "broken", True),
        ("broken-refused-after-text", "broken", True),
        ("broken-queued-after-started", "broken", False),
        ("broken-malformed", "broken", False),
        ("broken-no-terminal", "broken", True),
        ("rejected-rate-limited", "rejected:rate_limited", False),
        ("rejected-origin", "rejected:origin_not_allowed", False),
    ],
)
def test_each_example_grades_as_the_evals_expect(name: str, outcome: str, streamed_then_failed: bool):
    transcript = replay(name)
    assert (transcript.outcome, transcript.streamed_then_failed) == (outcome, streamed_then_failed)


def test_time_to_first_text_counts_from_the_request_start():
    # The clock ticks 100 ms per event read. The heartbeat is not an event, so the first text is the fifth.
    assert replay("answered").first_text_ms == 500
    assert replay("refused-pii").first_text_ms is None


def test_a_rejection_whose_status_disagrees_with_its_code_is_broken():
    body = (STREAMS / EXPECTED["rejected-rate-limited"]["file"]).read_text(encoding="utf-8")
    assert transcript_of_response(503, [body], clock=ticking_clock(), started=0.0).outcome == "broken"
    assert transcript_of_response(429, [body], clock=ticking_clock(), started=0.0).outcome == "rejected:rate_limited"


def test_a_non_contract_error_body_is_broken():
    transcript = transcript_of_response(502, ["<html>Bad Gateway</html>"], clock=ticking_clock(), started=0.0)
    assert transcript.outcome == "broken"


def test_every_stream_event_type_appears_in_some_example():
    shown = {type(frame) for f in FIXTURES if isinstance(f, StreamFixture) for frame in f.frames}
    missing = set(get_args(StreamEvent)) - shown
    assert missing == set()


def test_every_example_outcome_is_in_the_outcome_vocabulary():
    assert {f.expected.outcome for f in FIXTURES} <= set(OUTCOMES)

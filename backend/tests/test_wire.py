from pathlib import Path

import pytest

from vitae.contract import DoneEvent, SessionEvent, Source, SourcesEvent, TextEvent
from vitae.fixtures import FIXTURES, Raw, StreamFixture
from vitae.wire import Malformed, encode_event, read_events

STREAMS = Path(__file__).parents[2] / "contract" / "streams"
STREAM_FIXTURES = [f for f in FIXTURES if isinstance(f, StreamFixture) and f.name != "broken-malformed"]


def test_an_event_is_one_named_frame_with_one_compact_data_line():
    event = SourcesEvent(sources=(Source(id="projects", title="Projects"),))
    assert encode_event(event) == (
        b"event: sources\n"
        b'data: {"type":"sources","sources":[{"id":"projects","title":"Projects","href":null}]}\n\n'
    )


def test_text_with_unicode_line_separators_survives_a_splitlines_reader():
    event = TextEvent(text="Café naïve\u0085line\nbreak. ")
    assert list(read_events(encode_event(event).decode().splitlines())) == [event]


@pytest.mark.parametrize("fixture", STREAM_FIXTURES, ids=lambda f: f.name)
def test_every_committed_stream_reads_back_as_its_events(fixture: StreamFixture):
    lines = (STREAMS / f"{fixture.name}.sse").read_text(encoding="utf-8").splitlines()
    assert list(read_events(lines)) == [frame for frame in fixture.frames if not isinstance(frame, Raw)]


def test_a_payload_that_is_not_an_event_reads_as_malformed():
    lines = (STREAMS / "broken-malformed.sse").read_text(encoding="utf-8").splitlines()
    assert list(read_events(lines))[-1] == Malformed('{"type":"text","text":')


def test_an_unknown_event_type_reads_as_malformed():
    assert list(read_events(["event: progress", 'data: {"type":"progress"}', ""])) == [
        Malformed('{"type":"progress"}')
    ]


def test_comments_are_skipped_and_a_frame_without_its_blank_line_is_dropped():
    lines = [
        ": keep-alive",
        "",
        'data: {"type":"session","session_id":"c2Vzc2lvbi1leGFtcGxlMQ","turns_left":3}',
        "",
        'data: {"type":"done","finish":"complete"}',
    ]
    assert list(read_events(lines)) == [SessionEvent(session_id="c2Vzc2lvbi1leGFtcGxlMQ", turns_left=3)]


def test_frames_written_by_encode_event_concatenate_into_a_readable_stream():
    events = [SessionEvent(session_id="c2Vzc2lvbi1leGFtcGxlMQ", turns_left=1), DoneEvent(finish="length")]
    stream = b"".join(map(encode_event, events)).decode()
    assert list(read_events(stream.splitlines())) == events

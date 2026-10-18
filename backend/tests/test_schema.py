from pathlib import Path

from vitae.schema import contract_schema, export

CONTRACT = Path(__file__).parents[2] / "contract"
GENERATED_BY_NODE = {"vitae-contract.ts"}


def files_under(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in GENERATED_BY_NODE
    }


def test_committed_contract_files_equal_a_fresh_export(tmp_path: Path):
    export(tmp_path)
    assert files_under(CONTRACT) == files_under(tmp_path), "contract/ is stale: run `npm run contract`"


def test_the_ui_gets_outcome_and_stream_event_as_named_closed_unions():
    schema = contract_schema()
    assert schema["$defs"]["Outcome"]["enum"] == [
        "answered",
        "availability",
        "broken",
        "refused:unrelated",
        "refused:blocked_topic",
        "refused:injection",
        "refused:pii",
        "refused:unsafe",
        "refused:insufficient_evidence",
        "refused:output_rule",
        "rejected:rate_limited",
        "rejected:busy",
        "rejected:unavailable",
        "rejected:invalid_request",
        "rejected:origin_not_allowed",
        "rejected:too_long",
        "rejected:session_limit",
    ]
    assert [ref["$ref"] for ref in schema["$defs"]["StreamEvent"]["oneOf"]] == [
        "#/$defs/SessionEvent",
        "#/$defs/QueuedEvent",
        "#/$defs/StartedEvent",
        "#/$defs/TextEvent",
        "#/$defs/SourcesEvent",
        "#/$defs/DoneEvent",
        "#/$defs/RefusedEvent",
        "#/$defs/AvailabilityEvent",
        "#/$defs/ErrorEvent",
    ]
    assert schema["$defs"]["ErrorResponse"]["properties"]["error"] == {"$ref": "#/$defs/Rejection"}

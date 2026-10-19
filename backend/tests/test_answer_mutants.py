import pytest

from vitae.answer import MAX_SENTENCE_CHARS, Clean, OutputRules, OutputVerdict, Releaser, Violation, link_key
from vitae.contract import DoneEvent, ErrorEvent, RefusedEvent, Source, SourcesEvent, StreamEvent, TextEvent
from vitae.generation import Delta, Failed, Finished, GenEvent
from vitae.guards import RuleId

# The rules are built inside each test, not at import, so a mutant in OutputRules.build runs.
SOURCES = (Source(id="shift_notes", title="Shift Notes"),)


def rules() -> OutputRules:
    return OutputRules.build(
        "one two three four five six seven eight nine",
        allowed_links=("https://www.Alex.example/",),
        allowed_contacts=(" Hello@Alex.example ", "+1 (415) 555-0134"),
        allowed_names=("Alex Example",),
        refusal="I can't answer that one.",
    )


def test_build_keeps_the_keys_each_rule_compares():
    built = rules()
    assert built.allowed_links == frozenset({"alex.example"})
    assert built.allowed_contacts == frozenset({"hello@alex.example", "14155550134"})
    assert built.allowed_names == frozenset({frozenset({"alex", "example"})})
    assert built.refusal == "I can't answer that one."


@pytest.mark.parametrize(
    ("sentence", "verdict"),
    [
        ("Two three four five six seven eight nine. ", Violation(RuleId("output.echo"))),
        ("Three four five six seven eight nine. ", Clean()),
        ("One two three four five six seven. ", Clean()),
        ("See alex.example/work today. ", Clean()),
        ("See http://WWW.alex.example/x?y. ", Clean()),
        ("Call +1 415 555 0134 now. ", Clean()),
        ("Mail hello@alex.example now. ", Clean()),
        ("Dear Alex Example, hi. ", Clean()),
        ("Dear Bob Smith, hi. ", Violation(RuleId("output.name"))),
    ],
)
def test_freshly_built_rules_check_echo_links_contacts_and_names(sentence: str, verdict: OutputVerdict):
    assert rules().check(sentence) == verdict


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://www.Alex.example/path/),", "alex.example/path"),
        ("www.example.com//", "example.com"),
        ("(https://x.example", "(https://x.example"),
        ("/x.example/", "/x.example"),
        ("x.example/www.", "x.example/www"),
        ("https://WWW.x.example", "x.example"),
    ],
)
def test_link_key_drops_only_the_scheme_a_leading_www_and_trailing_punctuation(url: str, key: str):
    assert link_key(url) == key


def feed(chunks: list[str], *ends: GenEvent) -> tuple[list[StreamEvent], Releaser]:
    releaser = Releaser(rules(), SOURCES)
    events: list[StreamEvent] = []
    for chunk in chunks:
        events += releaser.feed(Delta(chunk))
    for end in ends:
        events += releaser.feed(end)
    return events, releaser


def test_a_fresh_releaser_has_not_finished_and_names_no_violation():
    releaser = Releaser(rules(), ())
    assert (releaser.finished, releaser.violation) == (False, None)


def test_a_sentence_of_exactly_the_limit_keeps_waiting():
    events, releaser = feed(["a" * MAX_SENTENCE_CHARS])
    assert (events, releaser.finished, releaser.violation) == ([], False, None)


def test_one_char_past_the_limit_is_refused_as_runaway():
    events, releaser = feed(["a" * (MAX_SENTENCE_CHARS + 1)])
    assert events == [RefusedEvent(reason="output_rule", message="I can't answer that one.")]
    assert (releaser.finished, releaser.violation) == (True, Violation(RuleId("output.runaway")))


def test_a_runaway_after_a_released_sentence_is_withheld():
    events, releaser = feed(["Hi there. Then", "a" * (MAX_SENTENCE_CHARS + 1)])
    assert events == [TextEvent(text="Hi there. "), DoneEvent(finish="withheld")]
    assert releaser.violation == Violation(RuleId("output.runaway"))


def test_nothing_is_released_after_a_complete_finish():
    events, releaser = feed(["Hi there. Bye"], Finished(finish="complete"), Delta("More text here. And"))
    assert events == [
        TextEvent(text="Hi there. "),
        TextEvent(text="Bye"),
        SourcesEvent(sources=SOURCES),
        DoneEvent(finish="complete"),
    ]
    assert releaser.released_text == "Hi there. Bye"


def test_nothing_is_released_after_a_length_finish():
    events, releaser = feed(["Hi there. Bye"], Finished(finish="length"), Delta("More text here. And"))
    assert events == [TextEvent(text="Hi there. "), SourcesEvent(sources=SOURCES), DoneEvent(finish="length")]
    assert releaser.released_text == "Hi there. "


def test_nothing_is_released_after_a_failure():
    events, releaser = feed(
        ["Hi there. Bye"], Failed(code="timeout"), Delta("More text here. And"), Finished(finish="complete")
    )
    assert events == [
        TextEvent(text="Hi there. "),
        ErrorEvent(code="timeout", detail="The model server did not finish in time."),
    ]
    assert releaser.finished is True

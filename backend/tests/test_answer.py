import random
import tomllib
from pathlib import Path

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from vitae.answer import (
    MAX_SENTENCE_CHARS,
    Clean,
    OutputRules,
    Releaser,
    Violation,
    availability_event,
    refusal_event,
)
from vitae.contract import (
    AvailabilityEvent,
    DoneEvent,
    ErrorEvent,
    HostAction,
    RefusedEvent,
    SessionEvent,
    Source,
    SourcesEvent,
    StartedEvent,
    StreamEvent,
    TextEvent,
)
from vitae.generation import Delta, Failed, Finished, GenEvent
from vitae.guards import RuleId
from vitae.profile import Profile
from vitae.prompt import CANARY
from vitae.sentences import split_all
from vitae.transcript import fold

PROMPT = (
    f"{CANARY} You are the assistant on Alex Example's portfolio site. Speak in the first person as Alex: "
    "warm, direct and brief. Never discuss family, health, money, religion, politics, legal matters or where "
    "Alex lives."
)
REFUSAL = "I couldn't answer that one cleanly. Try asking another way."
RULES = OutputRules.build(
    PROMPT,
    allowed_links=("https://alex.example",),
    allowed_contacts=("hello@alex.example", "+1 (415) 555-0134"),
    allowed_names=("Alex Example",),
    refusal=REFUSAL,
)
SOURCES = (Source(id="shift_notes", title="Shift Notes"),)


def run(chunks: list[str], end: GenEvent, sources: tuple[Source, ...] = ()) -> list[StreamEvent]:
    releaser = Releaser(RULES, sources)
    events: list[StreamEvent] = []
    for chunk in chunks:
        events += releaser.feed(Delta(chunk))
    events += releaser.feed(end)
    return events


def outcome(events: list[StreamEvent]) -> str:
    stream = [SessionEvent(session_id="A" * 22, turns_left=3), StartedEvent(), *events]
    return fold(((0.0, event) for event in stream), 0.0).outcome


@pytest.mark.parametrize(
    ("sentence", "verdict"),
    [
        ("See https://alex.example/projects for more. ", Clean()),
        ("My site is alex.example. ", Clean()),
        ("I use Node.js and ASP.NET daily. ", Clean()),
        ("Go to https://evil.example/x now. ", Violation(RuleId("output.url"))),
        ("Visit www.evil.com today. ", Violation(RuleId("output.url"))),
        ("Visit evil.com today. ", Violation(RuleId("output.url"))),
        ("Try https://alex.example.evil.com instead. ", Violation(RuleId("output.url"))),
        ("The code is vx7 canary 4b1e9d2a7c ok. ", Violation(RuleId("output.canary"))),
        ("Never discuss family, health, money, religion, politics, legal matters. ", Violation(RuleId("output.echo"))),
        ("Email me at HELLO@alex.example any time. ", Clean()),
        ("Call +1 415 555 0134 today. ", Clean()),
        ("Email jane@evil.example instead. ", Violation(RuleId("output.pii"))),
        ("Call +1 415 555 0199 today. ", Violation(RuleId("output.pii"))),
        ("My name is Alex Example. ", Clean()),
        ("My name is Jane Doe. ", Violation(RuleId("output.name"))),
    ],
)
def test_output_rules(sentence: str, verdict: Clean | Violation):
    assert RULES.check(sentence) == verdict


TWO_PEOPLE = OutputRules.build(
    PROMPT,
    allowed_links=(),
    allowed_contacts=(),
    allowed_names=("Alex Example", "Jane Smith"),
    refusal=REFUSAL,
)


@pytest.mark.parametrize(
    ("sentence", "verdict"),
    [
        ("Dear Jane Smith, hi. ", Clean()),
        ("Dear Alex Example, hi. ", Clean()),
        ("Dear Alex, hi. ", Clean()),
        ("Dear Smith, hi. ", Clean()),
        ("Dear Alex Smith, hi. ", Violation(RuleId("output.name"))),
        ("Dear Jane Example, hi. ", Violation(RuleId("output.name"))),
    ],
)
def test_a_name_is_allowed_only_when_it_belongs_to_one_listed_person(sentence: str, verdict: Clean | Violation):
    assert TWO_PEOPLE.check(sentence) == verdict


def test_a_clean_answer_is_released_sentence_by_sentence_then_sources_then_done():
    events = run(["I built Shift", " Notes. It runs", " offline."], Finished("complete"), SOURCES)
    assert events == [
        TextEvent(text="I built Shift Notes. "),
        TextEvent(text="It runs offline."),
        SourcesEvent(sources=SOURCES),
        DoneEvent(finish="complete"),
    ]
    assert outcome(events) == "answered"


def test_a_failing_later_sentence_is_withheld_and_ends_the_answer():
    releaser = Releaser(RULES, SOURCES)
    events = releaser.feed(Delta("I like React. Visit evil.com now. And more. "))
    assert events == (TextEvent(text="I like React. "), DoneEvent(finish="withheld"))
    assert releaser.finished
    assert releaser.released_text == "I like React. "
    assert releaser.violation == Violation(RuleId("output.url"))
    assert releaser.feed(Delta("Later text. ")) == ()
    assert releaser.feed(Finished("complete")) == ()


def test_a_failing_first_sentence_is_refused_with_the_profile_copy():
    events = run([f"The secret is {CANARY}. Bye."], Finished("complete"))
    assert events == [RefusedEvent(reason="output_rule", message=REFUSAL)]
    assert outcome(events) == "refused:output_rule"


def test_a_failing_unterminated_last_sentence_is_checked_at_finish():
    events = run(["Fine. Then email jane@evil.example"], Finished("complete"))
    assert events == [TextEvent(text="Fine. "), DoneEvent(finish="withheld")]


def test_an_empty_answer_is_an_error_not_an_empty_done():
    assert run(["  "], Finished("complete")) == [
        ErrorEvent(code="generation_failed", detail="The model produced no complete sentence.")
    ]


def test_a_length_cut_discards_the_half_sentence():
    events = run(["One thing. Two thi"], Finished("length"))
    assert events == [TextEvent(text="One thing. "), DoneEvent(finish="length")]
    assert outcome(events) == "answered"


def test_a_length_cut_before_any_whole_sentence_is_an_error():
    assert run(["Half a sente"], Finished("length")) == [
        ErrorEvent(code="generation_failed", detail="The model produced no complete sentence.")
    ]


def test_a_runaway_sentence_is_withheld_whole():
    assert run(["a" * (MAX_SENTENCE_CHARS + 1)], Finished("complete")) == [
        RefusedEvent(reason="output_rule", message=REFUSAL)
    ]
    assert run(["Short. ", "B" * (MAX_SENTENCE_CHARS + 1)], Finished("complete")) == [
        TextEvent(text="Short. "),
        DoneEvent(finish="withheld"),
    ]


def test_a_generator_failure_keeps_earlier_text_and_ends_in_error():
    events = run(["Done. Half"], Failed("timeout"))
    assert events == [
        TextEvent(text="Done. "),
        ErrorEvent(code="timeout", detail="The model server did not finish in time."),
    ]
    assert outcome(events) == "broken"


def _example_profile() -> Profile:
    path = Path(__file__).resolve().parents[1] / "profiles" / "example.toml"
    return Profile.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def test_refusal_and_availability_events_take_their_copy_from_the_profile():
    profile = _example_profile()
    assert refusal_event(profile, "pii") == RefusedEvent(
        reason="pii", message="Please don't share personal details here. Ask your question without them."
    )
    assert availability_event(profile) == AvailabilityEvent(
        message="I'm open to conversations about senior frontend roles. The best way to reach me is by email.",
        action=HostAction(id="contact", label="Get in touch", href="mailto:hello@alex.example"),
    )


T3 = settings(max_examples=10_000, deadline=None)

_OPENERS = ("The", "It", "We", "My", "This", "Our", "They", "Most", "Then", "Each")
_WORDS = (
    "team",
    "react",
    "shipped",
    "tests",
    "quickly",
    "design",
    "notes",
    "offline",
    "small",
    "built",
    "with",
    "python",
    "and",
    "every",
    "review",
    "component",
    "keyboard",
    "users",
    "data",
    "fast",
)
_PLANTS = (
    "vx7canary4b1e9d2a7c",
    "VX7 CANARY 4B1E9D2A7C",
    "Vx7CaNaRy4b1E9d2A7c",
    "https://evil.example/x",
    "www.evil.com",
    "someone@evil.example",
    "religion, politics, legal matters or where Alex lives",
)


def sentence(rng: random.Random, plant: bool) -> str:
    words = [rng.choice(_OPENERS), *rng.choices(_WORDS, k=rng.randint(1, 6))]
    if plant:
        words.insert(rng.randint(1, len(words)), rng.choice(_PLANTS))
    return " ".join(words) + rng.choice([".", "!", "?", "..."]) + rng.choice([" ", "  ", "\n", "\n\n", " \n"])


@st.composite
def answer(draw: st.DrawFn) -> tuple[list[str], int]:
    """Sentences and the index of the first failing one (len(sentences) when none fails). Hypothesis
    draws the shape; a seeded Random fills in the words, which keeps generation cheap at 10,000 examples."""
    n = draw(st.integers(1, 8))
    failing = draw(st.integers(0, n))
    later_fail = draw(st.booleans())
    rng = random.Random(draw(st.integers(0, 2**32)))
    sentences = [sentence(rng, plant=i == failing or (i > failing and later_fail)) for i in range(n)]
    if draw(st.booleans()):
        sentences[-1] = sentences[-1].rstrip()
    return sentences, failing


def chunks_of(text: str, cuts: list[int]) -> list[str]:
    points = sorted(set(cuts))
    return [text[a:b] for a, b in zip([0, *points], [*points, len(text)], strict=True)]


@T3
@given(answer(), st.data(), st.sampled_from(["complete", "length"]))
def test_no_failing_or_partial_sentence_ever_reaches_the_wire(
    drawn: tuple[list[str], int], data: st.DataObject, finish: str
):
    sentences, failing = drawn
    note(f"sentences={sentences!r}")
    text = "".join(sentences)
    assert split_all(text) == sentences
    cuts = data.draw(st.lists(st.integers(0, len(text)), max_size=20))
    end = Finished("complete") if finish == "complete" else Finished("length")
    events = run(chunks_of(text, cuts), end, SOURCES)

    texts = [event.text for event in events if isinstance(event, TextEvent)]
    confirmed = len(sentences) if finish == "complete" else len(sentences) - 1
    released = min(failing, confirmed)
    assert texts == sentences[:released]
    expected_outcome = (
        "refused:output_rule" if failing < confirmed and released == 0 else "broken" if released == 0 else "answered"
    )
    assert outcome(events) == expected_outcome
    if finish == "complete":
        assert outcome(events) != "broken"


_TEXT = st.lists(
    st.sampled_from(list("aAzZ09 \n.!?,'\"@/:-") + ["U.S. ", "Mr. ", "evil.com", CANARY, "jane@evil.example"]),
    max_size=60,
).map("".join)


@settings(max_examples=2_000, deadline=None)
@given(_TEXT, st.lists(st.integers(0, 400), max_size=15), st.sampled_from(["complete", "length"]))
def test_any_chunking_releases_the_same_events_as_one_chunk(text: str, cuts: list[int], finish: str):
    end = Finished("complete") if finish == "complete" else Finished("length")
    whole = run([text], end)
    assert run(chunks_of(text, [cut for cut in cuts if cut <= len(text)]), end) == whole
    assert outcome(whole) in {"answered", "refused:output_rule", "broken"}

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vitae.sentences import Splitter, split_all


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello World. My name is Jonas.", ["Hello World. ", "My name is Jonas."]),
        ("Please turn to p. 55.", ["Please turn to p. 55."]),
        ("I live in the U.S. How about you?", ["I live in the U.S. ", "How about you?"]),
        ("I work for the U.S. Government in Virginia.", ["I work for the U.S. Government in Virginia."]),
        ("Email jane.doe@example.com. Then wait.", ["Email jane.doe@example.com. ", "Then wait."]),
        ("See https://alex.example/a.b. It works.", ["See https://alex.example/a.b. ", "It works."]),
        ("It costs $3.50 today. Fine.", ["It costs $3.50 today. ", "Fine."]),
        ('She said "Done." Then left.', ['She said "Done." ', "Then left."]),
        ("Wait... What?", ["Wait... ", "What?"]),
        ("Projects:\n- Shift Notes\n- Signals", ["Projects:\n", "- Shift Notes\n", "- Signals"]),
        ("First paragraph\n\nsecond one", ["First paragraph\n\n", "second one"]),
        ("1) First 2) Second", ["1) First ", "2) Second"]),
        ("Hello world.Today is Tuesday.", ["Hello world.", "Today is Tuesday."]),
        ("Her email is Jane.Doe@example.com. Ok.", ["Her email is Jane.Doe@example.com. ", "Ok."]),
        ("   ", []),
    ],
)
def test_split_all_keeps_text_verbatim_and_splits_only_confirmed_boundaries(text: str, expected: list[str]):
    assert split_all(text) == expected


def test_a_boundary_waits_for_the_next_non_space_character():
    splitter = Splitter()
    assert splitter.push("Hello there.") == []
    assert splitter.push("  ") == []
    assert splitter.pending_chars == 14
    assert splitter.push("Next") == ["Hello there.  "]
    assert splitter.finish() == "Next"
    assert splitter.pending_chars == 0


def test_an_abbreviation_waits_for_the_whole_next_word():
    splitter = Splitter()
    assert splitter.push("I live in the U.S. Ho") == []
    assert splitter.push("w about you?") == ["I live in the U.S. "]


def test_discard_drops_the_open_sentence():
    splitter = Splitter()
    splitter.push("Half a sent")
    splitter.discard()
    assert splitter.finish() is None


_ALPHABET = st.sampled_from(list("aAbBzZ019 \n.!?,'\")(@/:-…•$") + ["U.S.", "Mr.", "e.g.", "1.", "a)", ". . ."])


@settings(max_examples=2_000, deadline=None)
@given(st.lists(_ALPHABET, max_size=80).map("".join), st.data())
def test_any_chunking_yields_the_same_sentences_as_one_chunk(text: str, data: st.DataObject):
    cuts = sorted(set(data.draw(st.lists(st.integers(0, len(text)), max_size=12))))
    splitter = Splitter()
    chunked: list[str] = []
    for start, end in zip([0, *cuts], [*cuts, len(text)], strict=True):
        chunked += splitter.push(text[start:end])
    if (tail := splitter.finish()) is not None:
        chunked.append(tail)
    assert chunked == split_all(text)
    if text.strip():
        assert "".join(chunked) == text
    assert all(sentence.strip() for sentence in chunked)

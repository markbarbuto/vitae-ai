import pytest

from vitae.sentences import Splitter, split_all


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("He left. -Then she came.", ["He left. ", "-Then she came."]),
        ("It works. 3 more to go.", ["It works. ", "3 more to go."]),
        ("Wait. «Quoted» text.", ["Wait. ", "«Quoted» text."]),
        ("So. a lowercase.", ["So. a lowercase."]),
        ("Hi. #Tag next.", ["Hi. ", "#Tag next."]),
        ("Hi. ● dot", ["Hi. ", "● dot"]),
        ("Hi. ü small", ["Hi. ü small"]),
        ("Hi. Ü big", ["Hi. ", "Ü big"]),
        ("Hi. ? weird", ["Hi. ? weird"]),
        ("Really?! Yes.", ["Really?! ", "Yes."]),
        ("What... No way.", ["What... ", "No way."]),
        ("Wait.. Then go.", ["Wait.. ", "Then go."]),
        ("Really! . . . The end.", ["Really! . . . The end."]),
        ("Wait.. . . . The end.", ["Wait.. . . . The end."]),
        ("(Hello.) Then more.", ["(Hello.) ", "Then more."]),
        ("**Bold.** Next.", ["**Bold.** ", "Next."]),
    ],
)
def test_a_terminator_ends_a_sentence_only_before_a_sentence_start(text: str, sentences: list[str]):
    assert split_all(text) == sentences


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("Hi.\nnext line", ["Hi.\nnext line"]),
        ("A line\nnext lower", ["A line\nnext lower"]),
        ("a line\nnext lower", ["a line\n", "next lower"]),
        ("Line one\nLine two", ["Line one\n", "Line two"]),
        ("first line\n\nsecond para", ["first line\n\n", "second para"]),
        ("42\nthe answer", ["42\n", "the answer"]),
        ("The cat\nthe dog", ["The cat\nthe dog"]),
        ("1. the cat\nthe dog", ["1. the cat\n", "the dog"]),
    ],
)
def test_a_line_break_joins_a_wrapped_capitalised_sentence_and_splits_lowercase_fragments(
    text: str, sentences: list[str]
):
    assert split_all(text) == sentences


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("End . Next one.", ["End . ", "Next one."]),
        ("word. . . . The end.", ["word. ", ". . . The end."]),
        ("word . . . The end.", ["word . . . The end."]),
        ("word . . . . The end.", ["word . . . . ", "The end."]),
        (". . . The end.", [". . . The end."]),
        (". . . . The end.", [". . . . ", "The end."]),
        ("a.a. . . The end.", ["a.a. . . ", "The end."]),
        ("e.g. . . The end.", ["e.g. . . ", "The end."]),
        ("compounds. . . the rest.", ["compounds. . . the rest."]),
        ("compounds. . . . the rest.", ["compounds. . . . the rest."]),
        ("a. . The", ["a. . ", "The"]),
        ("End. .x", ["End. .x"]),
        ("Go. ... Next", ["Go. ... ", "Next"]),
        ("  . Then", ["  . ", "Then"]),
    ],
)
def test_a_spaced_ellipsis_stays_inside_its_sentence_unless_a_period_ends_it(text: str, sentences: list[str]):
    assert split_all(text) == sentences


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("Mr. Smith came. He left.", ["Mr. Smith came. ", "He left."]),
        ("(Mr. Smith) came. He left.", ["(Mr. Smith) came. ", "He left."]),
        ("(J. 5) here.", ["(J. 5) here."]),
        ("Acme Inc. The company.", ["Acme Inc. ", "The company."]),
        ("Acme Inc. is big.", ["Acme Inc. is big."]),
        ("Acme Inc. Foo is big.", ["Acme Inc. Foo is big."]),
        ("Plan B. The next.", ["Plan B. ", "The next."]),
        ("U.S. Then later.", ["U.S. ", "Then later."]),
        ("J. Smith came.", ["J. Smith came."]),
        ("No. 5 is here.", ["No. 5 is here."]),
        ("St. Louis is big.", ["St. Louis is big."]),
        ("1. First item here.", ["1. First item here."]),
        ("• 1. Item. Then", ["• 1. Item. ", "Then"]),
    ],
)
def test_an_abbreviation_ends_a_sentence_only_before_a_usual_opener(text: str, sentences: list[str]):
    assert split_all(text) == sentences


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("world.Today we go", ["world.", "Today we go"]),
        ("Hello World.Ok then", ["Hello World.", "Ok then"]),
        ("a,b.Today is good", ["a,b.", "Today is good"]),
        ("world.today we go", ["world.today we go"]),
        ("Inc.Today we go", ["Inc.Today we go"]),
        ("x.Today", ["x.Today"]),
        ("Dr.Who is here. Ok.", ["Dr.Who is here. ", "Ok."]),
    ],
)
def test_a_period_glued_to_a_capitalised_word_splits_outside_addresses_and_abbreviations(
    text: str, sentences: list[str]
):
    assert split_all(text) == sentences


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("1) The first item 2) The second item", ["1) The first item ", "2) The second item"]),
        ("1) first 2) second", ["1) first ", "2) second"]),
        ("1) The first 3) The third", ["1) The first 3) The third"]),
        ("1. first 2) second", ["1. first 2) second"]),
        ("a) first b) second", ["a) first ", "b) second"]),
        ("a. first b. second", ["a. first ", "b. second"]),
        ("a) first c) third", ["a) first c) third"]),
        ("a) first b. second", ["a) first b. second"]),
        ("B) first c) second", ["B) first c) second"]),
        ("• first • second", ["• first ", "• second"]),
        ("•Apples •Pears", ["•Apples ", "•Pears"]),
        ("• first ◦ second", ["• first ◦ second"]),
        ("1) Python\nRust", ["1) Python\n", "Rust"]),
        ("1) Go\nOk", ["1) Go\n", "Ok"]),
        ("1) first 2", ["1) first 2"]),
    ],
)
def test_an_inline_list_splits_before_the_next_marker_of_the_same_kind(text: str, sentences: list[str]):
    assert split_all(text) == sentences


def test_streamed_chunks_release_each_sentence_once_its_end_is_confirmed():
    splitter = Splitter()
    assert [splitter.push(chunk) for chunk in ["Hel", "lo. ", "Wor", "ld. Th", "is"]] == [
        [],
        [],
        ["Hello. "],
        ["World. "],
        [],
    ]
    assert splitter.finish() == "This"


def test_a_list_marker_split_across_chunks_waits_for_its_lookahead():
    splitter = Splitter()
    assert [splitter.push(chunk) for chunk in ["1) first 2", ") second"]] == [[], ["1) first "]]
    assert splitter.finish() == "2) second"


def test_leading_whitespace_stays_with_the_first_sentence():
    assert split_all(" \nThe start. Then more") == [" \nThe start. ", "Then more"]


def test_a_splitter_reused_after_discard_starts_like_a_new_one():
    splitter = Splitter()
    splitter.push("Hello there. Wor")
    splitter.discard()
    assert splitter.pending_chars == 0
    assert splitter.push(" \nThe start. Then more") == [" \nThe start. "]
    assert splitter.finish() == "Then more"

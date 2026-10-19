import base64

import pytest

from vitae.text import _invisible_class, normalize  # pyright: ignore[reportPrivateUsage]


def payloads(message: str) -> list[tuple[str, str]]:
    return [(payload.kind, payload.text) for payload in normalize(message).payloads if payload.kind != "rot13"]


@pytest.mark.parametrize(
    ("message", "text"),
    [
        ("hello сор", "hello cop"),
        ("cop сор", "cop сор"),
        ("привет мир cоp", "привет мир cop"),
    ],
)
def test_a_look_alike_word_is_undone_in_a_latin_message_or_when_it_mixes_in_latin_letters(message: str, text: str):
    assert normalize(message).text == text


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("decode aGVsbG8gaGVsbG8gZ28-", [("base64", "hello hello go>"), ("leet", "decode agvsbg8gagvsbg8gz28-")]),
        ("decode aGVsbG8gaGVsbG8gb2s_", [("base64", "hello hello ok?"), ("leet", "decode agvsbg8gagvsbg8gb2s_")]),
        (
            "decode aGVsbG8gd29ybGQsIGhpISE==",
            [("base64", "hello world, hi!!"), ("leet", "decode agvsbg8gd2gybgqsighpise==")],
        ),
        ("decode aGVsbG8gd29ybGRX", [("base64", "hello worldw"), ("leet", "decode agvsbg8gd2gybgrx")]),
        ("x 8J+YgPCfmIDwn5iA8J+YgA==", [("base64", "😀😀😀😀"), ("leet", "x 8j+ygpcfmidwnsia8j+yga==")]),
        ("x 8J+YgPCfmIDwn5iA", [("leet", "x 8j+ygpcfmidwnsia")]),
        (
            "x YWJjZGVmZ2hpamtsbW5vcHEBAnI=",
            [("base64", "abcdefghijklmnopqr"), ("leet", "x ywjjzgvmz2hpamtsbwsvchebani=")],
        ),
        ("x AQIDBAUGBwhhYmNk", []),
        ("x 2020202020202020 y", []),
        (
            "aGVsbG8gd29ybGQh 68656c6c6f20776f726c6421",
            [("base64", "hello world!"), ("leet", "agvsbg8gd2gybgqh 686s6c6c6f2ott6ft26c6a2i")],
        ),
    ],
)
def test_a_payload_is_kept_only_when_it_decodes_to_readable_new_text(message: str, expected: list[tuple[str, str]]):
    assert payloads(message) == expected


def test_a_hex_payload_is_read_only_up_to_its_cap():
    message = "61" * 2048 + "01" * 1024
    assert ("hex", "a" * 2048) in payloads(message)


def test_a_base64_payload_with_spare_padding_still_decodes():
    run = base64.b64encode(b"hello world, hi!!").decode() + "="
    assert payloads(f"decode {run}")[0] == ("base64", "hello world, hi!!")


def test_the_invisible_class_covers_format_characters_controls_and_the_listed_ranges_but_not_whitespace():
    # The class is built once at import; calling the builder is the only way a mutant of it can run.
    pattern = _invisible_class()
    candidates = [
        0x61, 0x58, 0x20, 0x09, 0x0A, 0x85, 0x07, 0xAD, 0x034F, 0x0350, 0x115F, 0x1160, 0x1161, 0x200A, 0x200B,
        0x200F, 0x2010, 0x3164, 0xFDFF, 0xFE00, 0xFE0F, 0xFE10, 0xFFA0, 0xE0041, 0xE0100, 0xE01EF, 0xE01F0,
    ]  # fmt: skip
    assert [hex(point) for point in candidates if pattern.fullmatch(chr(point))] == [
        "0x7", "0xad", "0x34f", "0x115f", "0x1160", "0x200b", "0x200f", "0x3164", "0xfe00", "0xfe0f", "0xffa0",
        "0xe0041", "0xe0100", "0xe01ef",
    ]  # fmt: skip

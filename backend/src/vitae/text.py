"""Text as language: one normalization per message, read by every input guard."""

import base64
import binascii
import codecs
import re
import sys
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import unquote

PayloadKind = Literal["base64", "hex", "url", "leet", "rot13"]


@dataclass(frozen=True, slots=True)
class Payload:
    kind: PayloadKind
    """How the payload was read out of the message. Encodings are explicit; leet and rot13 are guesses."""
    text: str
    """Folded, like Turn.folded."""
    lines: tuple[str, ...] = ()
    """Each line of a decoding that spans several lines, folded, grouped like Turn.lines."""


@dataclass(frozen=True, slots=True)
class Turn:
    """One visitor message, normalized once. Every guard reads these views; none re-normalizes."""

    text: str
    """NFKC, control and zero-width characters removed, whitespace collapsed, disguised words undone
    (look-alike letters inside Latin words, letters spelled out with spaces). What the model and classifier see."""
    folded: str
    """text casefolded, accents stripped, look-alike letters mapped to ASCII. What word rules match against."""
    squashed: str
    """folded with everything but letters and digits removed. Catches "i g n o r e" and "ig.no.re"."""
    lines: tuple["Turn", ...]
    """Each line of a message written on several lines, normalized on its own; empty for one line. An
    injection on a line of its own reads as one, where scored with the rest of the message it is diluted.
    At most MAX_LINES: longer messages are grouped into that many runs of consecutive lines."""
    payloads: tuple[Payload, ...]
    """Folded base64, hex, URL-escaped, leetspeak and rot13 readings of text, each with its kind."""

    @property
    def decoded(self) -> tuple[str, ...]:
        """The payload texts. Rules run on these."""
        return tuple(payload.text for payload in self.payloads)


MAX_PAYLOADS: Final = 4
MAX_LINES: Final = 8
MAX_PAYLOAD_CHARS: Final = 2048

_INVISIBLE_RANGES: Final = (
    (0x034F, 0x034F),  # combining grapheme joiner
    (0x115F, 0x1160),  # Hangul fillers render as nothing
    (0x3164, 0x3164),
    (0xFFA0, 0xFFA0),
    (0xFE00, 0xFE0F),  # variation selectors carry hidden bytes in "emoji smuggling"
    (0xE0100, 0xE01EF),
)
_WHITESPACE_CONTROLS: Final = frozenset("\t\n\v\f\r\x1c\x1d\x1e\x1f\x85")
_SPACES: Final = re.compile(r"\s+")
_ASCII_LETTER: Final = re.compile(r"[A-Za-z]")
_NON_ASCII: Final = re.compile(r"[^\x00-\x7f]")
_NON_ASCII_RUN: Final = re.compile(r"[^\x00-\x7f]+")
"""Only these can be combining marks or non-Latin letters, so the per-character work skips ASCII."""
_NOT_ALNUM: Final = re.compile(r"[\W_]+")
"""Exactly what str.isalnum rejects: re's word class is isalnum plus the underscore."""

_LOOK_ALIKES: Final = {
    "A": "АΑᎪ",
    "a": "аɑα",
    "B": "ВΒᏴ",
    "b": "ЬƄ",
    "C": "СϹᏟ",
    "c": "сϲ",
    "D": "ᎠⅮ",
    "d": "ԁ",
    "E": "ЕΕᎬ",
    "e": "еєε",
    "F": "Ϝ",
    "G": "ԌᏀ",
    "g": "ɡ",
    "H": "НΗᎻ",
    "h": "һ",
    "I": "ІΙӀ",
    "i": "іıιɩ",
    "J": "ЈᎫ",
    "j": "јϳ",
    "K": "КΚᏦ",
    "k": "κ",
    "L": "ᏞⅬ",
    "l": "ӏ",
    "M": "МΜᎷ",
    "N": "Ν",
    "n": "ո",
    "O": "ОΟՕ",
    "o": "оοσօ",
    "P": "РΡᏢ",
    "p": "рρ",
    "q": "ԛ",
    "S": "ЅᏚ",
    "s": "ѕ",
    "T": "ТΤᎢ",
    "t": "τ",
    "U": "Ս",
    "u": "υս",
    "V": "ᏙⅤ",
    "v": "νѵ",
    "W": "ᎳԜ",
    "w": "ѡԝ",
    "X": "ХΧ",
    "x": "хχ",
    "Y": "ҮΥ",
    "y": "уγ",
    "Z": "ΖᏃ",
}
"""Cyrillic, Greek, Armenian and Cherokee letters drawn like Latin ones, keyed by the letter they imitate.
NFKC leaves them alone."""
_CONFUSABLES: Final = str.maketrans({source: latin for latin, sources in _LOOK_ALIKES.items() for source in sources})

_FOREIGN_LOOK_ALIKES: Final = frozenset(
    source
    for sources in _LOOK_ALIKES.values()
    for source in sources
    if not unicodedata.name(source).startswith("LATIN")
)
"""Look-alikes from other scripts. A word mixing these with Latin letters is a disguise, not a language."""
_WORD: Final = re.compile(r"\w+")
_SPELLED_OUT: Final = re.compile(r"(?<!\S)\w(?: \w){2,}(?!\S)")
_LEET: Final = str.maketrans("4310579@$", "aeiostgas")
_LEET_WORD: Final = re.compile(r"\b(?=\w*[a-z])\w*[0-9@$]")

_BASE64_RUN: Final = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX_RUN: Final = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){8,}|\b(?:[0-9A-Fa-f]{2}[ :]?){8,}")
_URL_ESCAPE: Final = re.compile(r"%[0-9A-Fa-f]{2}")
_LETTERS: Final = re.compile(r"[A-Za-z]")


def _invisible_class() -> re.Pattern[str]:
    """Format characters, controls other than whitespace, and _INVISIBLE_RANGES, as one character class
    with consecutive code points merged into ranges. A regex scan runs in C, and sre matches a few dozen
    ranges far faster than hundreds of single characters. Built from unicodedata at import, so it
    follows the interpreter's Unicode version."""
    points = {
        point
        for point in range(sys.maxunicode + 1)
        if unicodedata.category(chr(point)) in ("Cf", "Cc") and chr(point) not in _WHITESPACE_CONTROLS
    }
    points.update(point for low, high in _INVISIBLE_RANGES for point in range(low, high + 1))
    ranges: list[tuple[int, int]] = []
    for point in sorted(points):
        if ranges and ranges[-1][1] == point - 1:
            ranges[-1] = (ranges[-1][0], point)
        else:
            ranges.append((point, point))
    return re.compile("[" + "".join(f"{re.escape(chr(low))}-{re.escape(chr(high))}" for low, high in ranges) + "]")


_INVISIBLE: Final = _invisible_class()


def _clean(message: str) -> str:
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", _INVISIBLE.sub("", message))).strip()


def _undisguise(text: str) -> str:
    """Look-alike letters back to Latin in words that are disguised, and "i g n o r e" back to one word.
    A word is disguised when it mixes look-alikes with Latin letters, or is all look-alikes in a mostly
    Latin message. Real Cyrillic or Greek words are left alone."""
    if text.isascii():
        return _SPELLED_OUT.sub(_joined, text)
    ascii_letters = len(_ASCII_LETTER.findall(text))
    latin_message = ascii_letters > sum(char.isalpha() for char in _NON_ASCII.findall(text))

    def word(match: re.Match[str]) -> str:
        found = match.group()
        if found.isascii():
            return found
        foreign = [char in _FOREIGN_LOOK_ALIKES for char in found if char.isalpha() and not char.isascii()]
        if not foreign or not all(foreign):
            return found
        if latin_message or any(char.isascii() and char.isalpha() for char in found):
            return found.translate(_CONFUSABLES)
        return found

    return _SPELLED_OUT.sub(_joined, _WORD.sub(word, text))


def _joined(match: re.Match[str]) -> str:
    return match.group().replace(" ", "")


def _fold(text: str) -> str:
    if text.isascii():
        return text.casefold()
    decomposed = unicodedata.normalize("NFKD", text.translate(_CONFUSABLES).casefold())
    stripped = _NON_ASCII_RUN.sub(_without_marks, decomposed)
    return unicodedata.normalize("NFC", stripped).translate(_CONFUSABLES)


def _without_marks(match: re.Match[str]) -> str:
    return "".join(char for char in match.group() if not unicodedata.combining(char))


def _squash(folded: str) -> str:
    return _NOT_ALNUM.sub("", folded)


def _readable(raw: bytes) -> str | None:
    """raw as text when it decodes as UTF-8 and is mostly printable, so random letters that happen
    to be valid base64 or hex don't become payloads."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(char.isprintable() or char.isspace() for char in text)
    return text if len(text) >= 4 and printable >= 0.9 * len(text) else None


def _from_base64(run: str) -> str | None:
    body = run.rstrip("=")[: MAX_PAYLOAD_CHARS // 3 * 4]
    altchars = b"-_" if "-" in body or "_" in body else None
    try:
        raw = base64.b64decode(body + "=" * (-len(body) % 4), altchars=altchars, validate=True)
    except binascii.Error:
        return None
    return _readable(raw)


def _from_hex(run: str) -> str | None:
    digits = re.sub(r"\\x|[ :]", "", run)[: MAX_PAYLOAD_CHARS * 2]
    return _readable(bytes.fromhex(digits[: len(digits) - len(digits) % 2]))


def _candidates(text: str, folded: str) -> Iterator[tuple[PayloadKind, str | None]]:
    for match in _BASE64_RUN.finditer(text):
        yield "base64", _from_base64(match.group())
    for match in _HEX_RUN.finditer(text):
        yield "hex", _from_hex(match.group().strip())
    if _URL_ESCAPE.search(text):
        yield "url", unquote(text)
    if _LEET_WORD.search(folded):
        yield "leet", folded.translate(_LEET)
    if _LETTERS.search(folded):
        yield "rot13", codecs.encode(folded, "rot13")


def _decodings(text: str, folded: str) -> tuple[Payload, ...]:
    payloads: list[Payload] = []
    for kind, raw in _candidates(text, folded):
        if not raw:
            continue
        raw = raw[:MAX_PAYLOAD_CHARS]
        payload = _fold(_clean(raw))
        if payload and all(payload != seen.text for seen in payloads):
            payloads.append(Payload(kind, payload, _folded_lines(raw)))
            if len(payloads) == MAX_PAYLOADS:
                break
    return tuple(payloads)


def _grouped(lines: list[str]) -> tuple[str, ...]:
    """At most MAX_LINES runs of consecutive lines, joined by spaces; nothing for fewer than two lines."""
    if len(lines) < 2:
        return ()
    size = -(-len(lines) // MAX_LINES)
    return tuple(" ".join(lines[start : start + size]) for start in range(0, len(lines), size))


def _folded_lines(raw: str) -> tuple[str, ...]:
    lines = raw.splitlines()
    if len(lines) < 2:
        return ()
    return _grouped([folded for line in lines if (folded := _fold(_clean(line)))])


def _lines(message: str) -> tuple[Turn, ...]:
    lines = message.splitlines()
    if len(lines) < 2:
        return ()
    return tuple(normalize(line) for line in _grouped([line for line in lines if _clean(line)]))


def normalize(message: str) -> Turn:
    text = _undisguise(_clean(message))
    folded = _fold(text)
    return Turn(
        text=text, folded=folded, squashed=_squash(folded), lines=_lines(message), payloads=_decodings(text, folded)
    )

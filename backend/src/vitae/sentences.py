"""The incremental sentence splitter the output guard releases by.

Every split falls right after a run of whitespace, except a period glued to a capitalized word
("world.Today"). URLs, e-mail addresses and decimals have no whitespace after their dots, so they
are never cut. Splitting too little only delays release; splitting too much could cut a URL or a
contact detail away from the rule that checks it, so every doubtful case keeps the sentence whole.
"""

import re
from typing import Final

_GAP: Final = re.compile(r"\s+|(?<=[a-z0-9]\.)(?=[A-Z])")
_TERMINATORS: Final = ".!?…"
_CLOSERS: Final = "\"'”’)]»*_"
_ENDINGS: Final = frozenset(_TERMINATORS + _CLOSERS)
_BULLETS: Final = "•⁃‣◦▪●"
_OPENERS: Final = "\"'“‘([«*_" + _BULLETS
_STARTS: Final = frozenset("\"'“‘«*#>|`-–" + _BULLETS)

_PREPOSITIVE: Final = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "mx",
        "dr",
        "prof",
        "st",
        "mt",
        "ft",
        "gen",
        "rev",
        "hon",
        "capt",
        "lt",
        "col",
        "sgt",
        "gov",
        "sen",
        "rep",
        "pres",
        "supt",
        "messrs",
        "mme",
        "mlle",
    ]
)
"""Titles that always lead into the next word, so a period after them never ends a sentence."""
_ABBREVIATIONS: Final = frozenset(
    [
        "co",
        "corp",
        "inc",
        "ltd",
        "llc",
        "bros",
        "jr",
        "sr",
        "etc",
        "vs",
        "approx",
        "dept",
        "est",
        "fig",
        "figs",
        "no",
        "nos",
        "n°",
        "p",
        "pp",
        "vol",
        "vols",
        "ed",
        "eds",
        "al",
        "cf",
        "ca",
        "ch",
        "sec",
        "ibid",
        "jan",
        "feb",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "oct",
        "nov",
        "dec",
        "hr",
        "hrs",
        "mins",
        "yrs",
        "govt",
        "intl",
        "univ",
        "assn",
        "ave",
        "blvd",
        "rd",
    ]
)
"""Abbreviations that may end a sentence: they do only when a usual sentence opener follows."""
_NOT_GLUED: Final = _PREPOSITIVE | _ABBREVIATIONS
_STARTERS: Final = frozenset(
    [
        "A",
        "About",
        "After",
        "All",
        "Also",
        "Although",
        "An",
        "And",
        "Any",
        "Are",
        "As",
        "At",
        "Because",
        "Before",
        "Both",
        "But",
        "By",
        "Can",
        "Could",
        "Did",
        "Do",
        "Does",
        "Each",
        "Even",
        "Every",
        "First",
        "For",
        "From",
        "Have",
        "He",
        "Her",
        "Here",
        "His",
        "How",
        "However",
        "I",
        "If",
        "In",
        "Is",
        "It",
        "Its",
        "Just",
        "Let",
        "Many",
        "May",
        "Most",
        "My",
        "No",
        "Not",
        "Now",
        "Of",
        "On",
        "One",
        "Only",
        "Or",
        "Our",
        "Please",
        "She",
        "Should",
        "Since",
        "So",
        "Some",
        "Still",
        "That",
        "The",
        "Their",
        "Then",
        "There",
        "These",
        "They",
        "This",
        "Those",
        "To",
        "Today",
        "Was",
        "We",
        "Were",
        "What",
        "When",
        "Where",
        "Which",
        "While",
        "Who",
        "Why",
        "Will",
        "With",
        "Would",
        "Yes",
        "Yet",
        "You",
        "Your",
    ]
)
_MULTI_PERIOD: Final = re.compile(r"(?:[A-Za-z]{1,2}\.)+[A-Za-z]{1,2}")
_WORD: Final = re.compile(r"[A-Za-z]+")
_GLUED_LEFT: Final = re.compile(r"[A-Za-z0-9,]+")
_GLUED_RIGHT: Final = re.compile(r"[A-Z][a-z]+[.,!?;:\"'”’)]*")
_DOT_RUN: Final = re.compile(r"\.(?: \.)*")
_SPACE: Final = re.compile(r"\s+")
_LIST_MARKER: Final = re.compile(r"\d{1,3}|[A-Za-z]")
_OPENING_ITEM: Final = re.compile(
    r"\s*(?:(?P<bullet>["
    + _BULLETS
    + r"])|(?P<num>\d{1,3})(?P<np>\.\)|[.)])(?=\s)|(?P<let>[a-z])(?P<lp>\.\)|[.)])(?=\s))"
)
_NEXT_ITEM: Final = re.compile(r"(?:(?P<num>\d{1,3})|(?P<let>[a-z]))(?P<p>\.\)|[.)])\s")


def _starts(char: str) -> bool:
    return (char.isalpha() and not char.islower()) or char.isdigit() or char in _STARTS


class Splitter:
    """Incremental. push() returns only sentences whose end is confirmed by what follows, so a
    sentence waits for the first non-space character after its terminator (one token, typically).
    Trailing whitespace belongs to the sentence it follows, so released texts concatenate verbatim."""

    __slots__ = ("_buf", "_scan")

    def __init__(self) -> None:
        self._buf = ""
        self._scan = 0

    def push(self, chunk: str) -> list[str]:
        self._buf += chunk
        sentences: list[str] = []
        while (end := self._boundary()) is not None:
            sentences.append(self._buf[:end])
            self._buf = self._buf[end:]
            self._scan = 0
        return sentences

    def finish(self) -> str | None:
        """The unterminated rest, if it holds anything but whitespace. A boundary still waiting
        for lookahead at the end is not a boundary."""
        tail = self._buf
        self.discard()
        return tail if tail.strip() else None

    def discard(self) -> None:
        self._buf = ""
        self._scan = 0

    @property
    def pending_chars(self) -> int:
        return len(self._buf)

    def _boundary(self) -> int | None:
        buf = self._buf
        for gap in _GAP.finditer(buf, self._scan):
            start, end = gap.span()
            verdict = None if end == len(buf) else _decide(buf, start, end)
            if verdict is None:
                self._scan = start
                return None
            if verdict:
                return end
        self._scan = len(buf)
        return None


def split_all(text: str) -> list[str]:
    """push(text) then finish(). The T4 golden-rules harness calls this, so it scores the streaming splitter."""
    splitter = Splitter()
    sentences = splitter.push(text)
    if (tail := splitter.finish()) is not None:
        sentences.append(tail)
    return sentences


def _decide(buf: str, start: int, end: int) -> bool | None:
    """Whether the gap buf[start:end] ends the sentence that begins at buf[0]. None means the
    answer needs text that has not arrived yet."""
    if start == end:
        return _glued(buf, start)
    if start == 0:
        return False
    gap = buf[start:end]
    nxt = buf[end]
    if gap == " " and buf[start - 1] not in _ENDINGS:
        return _next_list_item(buf, end)
    if gap.count("\n") >= 2:
        return True
    if "\n" in gap and (not nxt.islower() or not _opens_upper(buf, start)):
        return True
    tok_start = _token_start(buf, start)
    tok = buf[tok_start:start]
    core = tok.rstrip(_CLOSERS)
    if not core or core[-1] not in _TERMINATORS:
        return _next_list_item(buf, end)
    if core[-1] != "." or core.endswith(".."):
        return _starts(nxt)
    if core == ".":
        return _spaced_dot(buf, tok_start, nxt)
    if nxt == ".":
        return _period_then_ellipsis(buf, end)
    word = core[:-1].lstrip(_OPENERS)
    if _LIST_MARKER.fullmatch(word) and all(char.isspace() or char in _BULLETS for char in buf[:tok_start]):
        return False
    key = word.casefold()
    if key in _PREPOSITIVE:
        return False
    if key in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()) or _MULTI_PERIOD.fullmatch(word):
        if not nxt.isupper():
            return False
        following = _WORD.match(buf, end)
        if following is None or following.end() == len(buf):
            return None
        return following.group() in _STARTERS
    return _starts(nxt)


def _token_start(buf: str, start: int) -> int:
    i = start
    while i > 0 and not buf[i - 1].isspace():
        i -= 1
    return i


def _opens_upper(buf: str, start: int) -> bool:
    """A line break before a lowercase word joins a wrapped line of a capitalized sentence, but
    splits a list of lowercase fragments."""
    for char in buf[:start]:
        if char.isalpha():
            return char.isupper()
    return False


def _glued(buf: str, at: int) -> bool | None:
    """A period with no space before a capitalized word, as in "world.Today". Never inside a
    token that looks like an address, a domain or an abbreviation."""
    left_start = _token_start(buf, at - 1)
    left = buf[left_start : at - 1]
    if not _GLUED_LEFT.fullmatch(left) or left.casefold() in _NOT_GLUED:
        return False
    right_end = at
    while right_end < len(buf) and not buf[right_end].isspace():
        right_end += 1
    if right_end == len(buf):
        return None
    return _GLUED_RIGHT.fullmatch(buf, at, right_end) is not None


def _spaced_dot(buf: str, tok_start: int, nxt: str) -> bool:
    """A lone dot in a spaced ellipsis. Three dots after a space are an ellipsis inside the
    sentence; four are an ellipsis then a period. "word. . . ." was split after "word."."""
    first = tok_start
    dots = 1
    while first >= 2 and buf[first - 1] == " " and buf[first - 2] == ".":
        dots += 1
        first -= 2
    attached = first > 0 and not buf[first - 1].isspace()
    if attached:
        return dots < 4 and _starts(nxt)
    return (dots == 1 or dots >= 4) and _starts(nxt)


def _period_then_ellipsis(buf: str, end: int) -> bool | None:
    """ "compounds. . . . The": the period ends the sentence and the ellipsis opens the next."""
    run = _DOT_RUN.match(buf, end)
    assert run is not None
    after = run.end()
    if after == len(buf):
        return None
    space = _SPACE.match(buf, after)
    if space is None:
        return False
    if space.end() == len(buf):
        return None
    return run.group().count(".") >= 3 and _starts(buf[space.end()])


def _next_list_item(buf: str, end: int) -> bool | None:
    """Inline lists: "1) The first item 2) The second item". Only a sentence that opened with an
    item marker splits before the next marker of the same kind."""
    if buf[0].isupper():
        return False
    opening = _OPENING_ITEM.match(buf)
    if opening is None:
        return False
    if (bullet := opening.group("bullet")) is not None:
        return buf[end] == bullet
    item = _NEXT_ITEM.match(buf, end)
    if item is None:
        rest = buf[end : end + 6]
        return None if len(rest) < 6 and not any(char.isspace() for char in rest) else False
    if (num := opening.group("num")) is not None:
        return item.group("num") == str(int(num) + 1) and item.group("p") == opening.group("np")
    letter = opening.group("let")
    return item.group("let") == chr(ord(letter) + 1) and item.group("p") == opening.group("lp")

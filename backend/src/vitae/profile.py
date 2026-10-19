"""The owner's profile file, parsed once at the boundary. The single source of refusal copy."""

import re
import tomllib
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Final, Self, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    RootModel,
    StringConstraints,
    ValidationError,
    model_validator,
)

from vitae.contract import HostAction, RefusalReason, Slug, Text
from vitae.text import normalize

GAP: Final = "*"
GAP_WORDS: Final = 4
MIN_STEM: Final = 3
"""Letters a `word*` needs before its star, so "a*" cannot match nearly every message."""


def term_words(term: str) -> tuple[str, ...]:
    """The term folded the way Turn.folded is, split into words. `*` alone is a gap, `word*` a prefix."""
    return tuple(normalize(term).folded.split())


def _check_term(term: str) -> str:
    words = term_words(term)
    if not words:
        raise ValueError("is blank")
    if words[0] == GAP or words[-1] == GAP:
        raise ValueError(f'"{term}": a lone * stands for up to {GAP_WORDS} words, so it goes between two words')
    for before, after in pairwise(words):
        if before == after == GAP:
            raise ValueError(f'"{term}": two lone * in a row')
    for word in words:
        stem = word.removesuffix(GAP)
        if GAP in stem:
            raise ValueError(f'"{term}": * goes on its own or at the end of a word')
        if word != GAP and word.endswith(GAP) and len(stem) < MIN_STEM:
            raise ValueError(f'"{term}": a word ending in * needs at least {MIN_STEM} letters before it')
    return term


Term = Annotated[str, StringConstraints(min_length=1, max_length=80), AfterValidator(_check_term)]
"""A word or phrase, matched as whole words on the folded message. A lone `*` stands for up to four
words and `word*` for any word starting with `word`. Nothing else is special, so an owner cannot write
a regex, or a ReDoS."""

_BETWEEN: Final = r"\W+"
_GAP_PATTERN: Final = rf"(?:\W+\w+){{0,{GAP_WORDS}}}\W+"


def _term_pattern(words: tuple[str, ...]) -> str:
    parts: list[str] = []
    for index, word in enumerate(words):
        if word == GAP:
            parts.append(_GAP_PATTERN)
            continue
        if index and words[index - 1] != GAP:
            parts.append(_BETWEEN)
        stem = word.removesuffix(GAP)
        parts.append(re.escape(stem) + (r"\w*" if word.endswith(GAP) else ""))
    return "".join(parts)


def compile_terms(terms: Iterable[str]) -> re.Pattern[str]:
    """One pattern matching any of the terms as whole words in folded text. \\W and \\w never overlap,
    so a gap can only split the text one way and matching stays linear."""
    alternatives = [_term_pattern(term_words(term)) for term in terms]
    if not alternatives:
        return re.compile(r"(?!)")
    return re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)")


class ProfileError(Exception):
    """Carries every problem at once, one `section.key: message` line per problem."""

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        noun = "problem" if len(problems) == 1 else "problems"
        super().__init__("\n  ".join((f"Invalid profile ({len(problems)} {noun}):", *problems)))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssistantSection(_Section):
    voice: Text
    rules: Text
    addressing: tuple[Term, ...]
    """Phrases that mean the visitor is asking about the owner. With an allowed term, one overrules
    the scope rule's off-scope shapes."""


class Topics(_Section):
    allowed: dict[Slug, tuple[Term, ...]]
    """Scope: an allowed term plus an addressing phrase overrules an off-scope shape, fuzzy for words of 5+ letters."""
    blocked: dict[Slug, tuple[Term, ...]]
    """Whole-word or phrase match on Turn.folded and Turn.decoded."""


class AvailabilitySection(_Section):
    when: tuple[Term, ...]
    answer: Text
    action: HostAction | None = None


class OutputSection(_Section):
    allowed_links: tuple[Text, ...]
    allowed_contacts: tuple[Text, ...]
    allowed_names: tuple[Text, ...]
    """Names an answer may use: the owner's, and anyone the owner allows the pack to name."""


class Refusals(RootModel[dict[RefusalReason, Text]]):
    """Total over RefusalReason: a validator compares keys with get_args(RefusalReason), so adding a
    reason to the contract fails every profile load until the owner writes its copy."""

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _total(self) -> Self:
        missing = [reason for reason in get_args(RefusalReason) if reason not in self.root]
        if missing:
            raise ValueError(f"missing copy for {', '.join(missing)}")
        return self

    def __getitem__(self, reason: RefusalReason) -> str:
        return self.root[reason]


class Suggestion(_Section):
    label: Text
    message: Text


class Profile(_Section):
    assistant: AssistantSection
    topics: Topics
    availability: AvailabilitySection
    output: OutputSection
    refusals: Refusals
    suggestions: tuple[Suggestion, ...]


def load_profile(path: Path) -> Profile:
    """The only reader of the file. Raises ProfileError listing every problem as `section.key: message`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProfileError((f"{path}: cannot be read ({error.strerror or error}).",)) from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ProfileError((f"{path}: not valid TOML ({error}).",)) from None
    try:
        return Profile.model_validate(data)
    except ValidationError as error:
        raise ProfileError(tuple(_problem(problem["loc"], problem["msg"]) for problem in error.errors())) from None


def _problem(loc: tuple[int | str, ...], message: str) -> str:
    where = ".".join(str(part) for part in loc if part != "[key]") or "profile"
    return f"{where}: {message.removeprefix('Value error, ')}"

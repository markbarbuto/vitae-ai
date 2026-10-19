"""The output rules and the sentence-buffered release. The only module that constructs a TextEvent,
and the only one that mints the profile's refusal and availability events."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Self, assert_never

from vitae.contract import (
    AvailabilityEvent,
    DoneEvent,
    ErrorEvent,
    Finish,
    RefusalReason,
    RefusedEvent,
    Source,
    SourcesEvent,
    StreamEvent,
    TextEvent,
)
from vitae.generation import Delta, Failed, Finished, GenEvent
from vitae.guards import RuleId
from vitae.guards.pii import PiiKind, find_pii
from vitae.profile import Profile
from vitae.prompt import CANARY
from vitae.sentences import Splitter

MAX_SENTENCE_CHARS: Final = 1000
"""A sentence still open past this is withheld as output.runaway, never released in part."""
SHINGLE_WORDS: Final = 8

EMPTY_DETAIL: Final = "The model produced no complete sentence."
FAILED_DETAIL: Final = {
    "generation_failed": "The model server stopped answering.",
    "timeout": "The model server did not finish in time.",
}

_CANARY_KEY: Final = CANARY.casefold()
_NOT_ALNUM: Final = re.compile(r"[\W_]+")
_WORDS: Final = re.compile(r"\w+")
_URL: Final = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://\S+"
    r"|\b(?i:www)\.\S+"
    r"|(?<![@\w.-])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|io|dev|ai|app|co|me|xyz|info|biz|edu|gov"
    r"|site|online|tech|ly|gg|tv|uk|us|ca|de|fr|au|nl|eu|ru|cn)\b(?:/\S*)?"
)
"""Any scheme, a www. host, or a bare domain on a common TLD. Bare domains need a lowercase TLD
listed here, so "Node.js" and "ASP.NET" are not links."""
_URL_TRAILING: Final = ".,;:!?)]}'\"”’>*_`"
_SCHEME: Final = re.compile(r"^[a-z][a-z0-9+.-]*://")


@dataclass(frozen=True, slots=True)
class Clean:
    pass


@dataclass(frozen=True, slots=True)
class Violation:
    rule: RuleId
    """"output.url", "output.canary", "output.echo", "output.pii", "output.name" or "output.runaway"."""


OutputVerdict = Clean | Violation

CLEAN: Final = Clean()


@dataclass(frozen=True, slots=True)
class OutputRules:
    """Built once at startup from the profile and the static system prompt."""

    allowed_links: frozenset[str]
    """From the profile, as link_key gives them. Phase 3 adds knowledge-pack URLs."""
    allowed_contacts: frozenset[str]
    """The owner's public email and phone from the profile, as contact_key gives them."""
    allowed_names: frozenset[str]
    """Casefolded words of the names the answer may use: the owner's, and anyone the profile lists."""
    prompt_shingles: frozenset[int]
    """Hashes of every 8-word shingle of the system prompt."""
    refusal: str
    """The profile's output_rule copy, sent when a violation comes before any released sentence."""

    @classmethod
    def build(
        cls,
        system_prompt: str,
        *,
        allowed_links: Iterable[str],
        allowed_contacts: Iterable[str],
        allowed_names: Iterable[str],
        refusal: str,
    ) -> Self:
        return cls(
            allowed_links=frozenset(link_key(link) for link in allowed_links),
            allowed_contacts=frozenset(contact_key(contact) for contact in allowed_contacts),
            allowed_names=frozenset(word for name in allowed_names for word in _WORDS.findall(name.casefold())),
            prompt_shingles=frozenset(_shingles(system_prompt)),
            refusal=refusal,
        )

    def check(self, sentence: str) -> OutputVerdict:
        """URL not in allowed_links; CANARY anywhere (case- and space-insensitive); any shingle of the
        sentence in prompt_shingles; find_pii(sentence) hits not in allowed contacts or names."""
        folded = unicodedata.normalize("NFKC", sentence).casefold()
        if _CANARY_KEY in _NOT_ALNUM.sub("", folded):
            return Violation(RuleId("output.canary"))
        for url in _URL.finditer(sentence):
            if not self._link_allowed(link_key(url.group())):
                return Violation(RuleId("output.url"))
        if not self.prompt_shingles.isdisjoint(_shingles(folded)):
            return Violation(RuleId("output.echo"))
        for hit in find_pii(sentence):
            value = sentence[hit.start : hit.end]
            if hit.kind is PiiKind.NAME:
                if not set(_WORDS.findall(value.casefold())) <= self.allowed_names:
                    return Violation(RuleId("output.name"))
            elif contact_key(value) not in self.allowed_contacts:
                return Violation(RuleId("output.pii"))
        return CLEAN

    def _link_allowed(self, key: str) -> bool:
        return any(
            key == allowed or key.startswith((f"{allowed}/", f"{allowed}?", f"{allowed}#"))
            for allowed in self.allowed_links
        )


def link_key(url: str) -> str:
    """Scheme, a leading www., trailing punctuation and a trailing slash dropped; casefolded.
    "https://alex.example/" and "alex.example" compare equal."""
    key = _SCHEME.sub("", url.casefold().rstrip(_URL_TRAILING))
    return key.removeprefix("www.").rstrip("/")


def contact_key(contact: str) -> str:
    """An e-mail casefolded, anything else reduced to its letters and digits, so phone
    formatting does not matter."""
    folded = contact.strip().casefold()
    return folded if "@" in folded else _NOT_ALNUM.sub("", folded)


def _shingles(text: str) -> Iterable[int]:
    words = _WORDS.findall(text.casefold())
    return (hash(tuple(words[i : i + SHINGLE_WORDS])) for i in range(len(words) - SHINGLE_WORDS + 1))


def refusal_event(profile: Profile, reason: RefusalReason) -> RefusedEvent:
    return RefusedEvent(reason=reason, message=profile.refusals[reason])


def availability_event(profile: Profile) -> AvailabilityEvent:
    return AvailabilityEvent(message=profile.availability.answer, action=profile.availability.action)


class Releaser:
    """The sentence-buffered release, as a sync state machine over GenEvents. The only place in the
    package that constructs a TextEvent (enforced by a ruff banned-api rule whose message names this class).
    One per turn; never shared."""

    __slots__ = ("_finished", "_released", "_rules", "_sources", "_splitter", "_violation")

    def __init__(self, rules: OutputRules, sources: tuple[Source, ...]) -> None:
        self._rules = rules
        self._sources = sources
        self._splitter = Splitter()
        self._released: list[str] = []
        self._finished = False
        self._violation: Violation | None = None

    def feed(self, event: GenEvent) -> tuple[StreamEvent, ...]:
        """Events to send, in order. Empty once finished."""
        if self._finished:
            return ()
        match event:
            case Delta(text=text):
                out: list[StreamEvent] = []
                for sentence in self._splitter.push(text):
                    out += self._admit(sentence)
                    if self._finished:
                        return tuple(out)
                if self._splitter.pending_chars > MAX_SENTENCE_CHARS:
                    out += self._end(Violation(RuleId("output.runaway")))
                return tuple(out)
            case Finished(finish="length"):
                self._splitter.discard()
                return self._end_ok("length")
            case Finished():
                tail = self._splitter.finish()
                out = list(self._admit(tail)) if tail is not None else []
                return tuple(out) if self._finished else (*out, *self._end_ok("complete"))
            case Failed(code=code):
                self._finished = True
                return (ErrorEvent(code=code, detail=FAILED_DETAIL[code]),)
            case _:
                assert_never(event)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def released_text(self) -> str:
        return "".join(self._released)

    @property
    def sentences_released(self) -> int:
        return len(self._released)

    @property
    def violation(self) -> Violation | None:
        """The rule that ended the turn, for the log record."""
        return self._violation

    def _admit(self, sentence: str) -> tuple[StreamEvent, ...]:
        match self._rules.check(sentence):
            case Clean():
                self._released.append(sentence)
                return (TextEvent(text=sentence),)
            case Violation() as violation:
                return self._end(violation)

    def _end(self, violation: Violation) -> tuple[StreamEvent, ...]:
        self._finished = True
        self._violation = violation
        if self._released:
            return (DoneEvent(finish="withheld"),)
        return (RefusedEvent(reason="output_rule", message=self._rules.refusal),)

    def _end_ok(self, finish: Finish) -> tuple[StreamEvent, ...]:
        self._finished = True
        if not self._released:
            return (ErrorEvent(code="generation_failed", detail=EMPTY_DETAIL),)
        sources = (SourcesEvent(sources=self._sources),) if self._sources else ()
        return (*sources, DoneEvent(finish=finish))

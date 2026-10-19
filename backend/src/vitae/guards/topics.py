"""Profile-driven rules: blocked topics, the availability answer, and the scope rule."""

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Final

from vitae.guards import PASS, Availability, Refuse, RuleId, Stage, Verdict
from vitae.profile import GAP, AvailabilitySection, Topics, compile_terms, term_words
from vitae.text import Turn

FUZZY_MIN_LETTERS: Final = 5
"""Words this long tolerate one edit (a typo). Shorter words tolerate only two letters swapped,
because one edit turns most short words into other real words."""
AVAILABILITY_RULE: Final = RuleId("availability")
TASK_RULE: Final = RuleId("scope.task")
TRIVIA_RULE: Final = RuleId("scope.trivia")
VISITOR_RULE: Final = RuleId("scope.visitor")

_WORD: Final = re.compile(r"\w+")
_IS: Final = r"(?:\s+is|['’]s|\s+was)"
_UP_TO_4_WORDS: Final = r"(?:\W+\w+){0,4}?"
_TASK_VERBS: Final = r"write|generate|create|compose|draft|produce|give|tell|make|send"
_TASK_OBJECTS: Final = (
    r"code|function|script|program|algorithm|regex|sql|essay|poem|story|stories|joke|song|lyrics|haiku"
    r"|limerick|letter|recipe|homework|assignment|equation|paragraph|article|blog post|tweet"
)
_START: Final = r"^\W*(?:(?:hey|hi|ok|okay|please)\W+)*(?:(?:can|could|would|will)\W+you\W+(?:please\W+)?)?"


def _shape(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w)(?:{pattern})(?!\w)")


OFF_SCOPE: Final = (
    (TASK_RULE, _shape(rf"(?:{_TASK_VERBS}){_UP_TO_4_WORDS}\W+(?:{_TASK_OBJECTS})")),
    (TASK_RULE, re.compile(rf"{_START}(?:translate|solve|compute|calculate|implement|debug|summari[sz]e)(?!\w)")),
    (TASK_RULE, _shape(r"homework|what(?:\s+is|['’]s)\s+\d+\s*[-+*/x×^]\s*\d+|solve\W+for\W+[a-z]")),
    (
        TRIVIA_RULE,
        _shape(
            rf"what{_IS}\s+the\s+(?:capital|population|currency|national \w+)\s+of"
            r"|who\s+(?:won|invented|discovered|painted)"
            rf"|how\s+(?:far|tall|high|deep|big|old){_IS}\s+the|weather\s+(?:in|like|today|tomorrow)"
        ),
    ),
    (
        VISITOR_RULE,
        _shape(
            r"should\s+i\s+(?:buy|sell|invest|learn|study|major|become|take|quit|switch|move|choose|pick|use)"
            r"|(?:do|would|will)\s+i\s+need|what\s+(?:should|do)\s+i\s+(?:study|learn|major|read|eat|cook|buy|watch)"
            r"|help\s+me\s+(?:with\s+my|write|fix|improve|prepare|study|choose|decide|learn|plan|pick)"
            r"|my\s+(?:resume|cv|essay|cover letter|assignment|thesis|dissertation)"
        ),
    ),
)
"""Engine-owned, high-precision shapes of a message that is not about the owner. Checked in order;
the first match names the rule."""
_REQUEST_FRAME: Final = re.compile(r"(?<!\w)(?:can|could|would|will|won['’]?t)\W+you(?!\w)")
""""Can you" frames a request, so its "you" does not make the message about the owner."""


def blocked_stage(topics: Topics) -> Stage:
    patterns = tuple((RuleId(f"topic.{slug}"), compile_terms(terms)) for slug, terms in topics.blocked.items())
    # One search over every term decides most texts; the per-topic loop then only names the rule.
    any_topic = compile_terms(term for terms in topics.blocked.values() for term in terms)

    def check(turn: Turn) -> Verdict:
        for text in (turn.folded, *turn.decoded):
            if any_topic.search(text) is None:
                continue
            for rule, pattern in patterns:
                if pattern.search(text):
                    return Refuse("blocked_topic", rule)
        return PASS

    return Stage("blocked_topic", check)


def availability_stage(availability: AvailabilitySection) -> Stage:
    pattern = compile_terms(availability.when)

    def check(turn: Turn) -> Verdict:
        return Availability(AVAILABILITY_RULE) if pattern.search(turn.folded) else PASS

    return Stage("availability", check)


def scope_stage(topics: Topics, addressing: tuple[str, ...]) -> Stage:
    """Refuses as unrelated only a message with a recognised off-scope shape: a general task, trivia,
    or the visitor's own life. A shape is overruled when the message also refers to the owner and
    names an allowed term ("write a summary of your projects"). Anything else passes."""
    owner = _Terms(addressing)
    allowed = _Terms(term for terms in topics.allowed.values() for term in terms)

    def check(turn: Turn) -> Verdict:
        shape = next((rule for rule, pattern in OFF_SCOPE if pattern.search(turn.folded)), None)
        if shape is None:
            return PASS
        unframed = _REQUEST_FRAME.sub(" ", turn.folded)
        words = frozenset(_WORD.findall(unframed))
        if owner.found(unframed, words) and allowed.found(turn.folded, frozenset(_WORD.findall(turn.folded))):
            return PASS
        return Refuse("unrelated", shape)

    return Stage("scope", check)


class _Terms:
    """Exact matching for every term, plus typo-tolerant matching for one-word terms."""

    def __init__(self, terms: Iterable[str]) -> None:
        terms = tuple(terms)
        self._exact = compile_terms(terms)
        self._plain: defaultdict[str, set[str]] = defaultdict(set)
        self._stems: defaultdict[int, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for words in map(term_words, terms):
            if len(words) != 1:
                continue
            term = words[0]
            stem = term.removesuffix(GAP)
            index = self._stems[len(stem)] if term.endswith(GAP) else self._plain
            for key in _deletions(stem):
                index[key].add(term)

    def found(self, folded: str, words: frozenset[str]) -> bool:
        return self._exact.search(folded) is not None or any(
            _near(word, term) for word in words for term in self._candidates(word)
        )

    def _candidates(self, word: str) -> set[str]:
        """The terms _near could accept for word. Any pair _one_typo accepts shares a string with at most
        one letter deleted from each, so looking those strings up skips the hopeless terms without
        calling _near on them."""
        found: set[str] = set()
        for key in _deletions(word):
            found.update(self._plain.get(key, ()))
        for length, index in self._stems.items():
            for prefix in {word[: length - 1], word[:length], word[: length + 1]}:
                for key in _deletions(prefix):
                    found.update(index.get(key, ()))
        return found


def _deletions(word: str) -> set[str]:
    return {word, *(word[:i] + word[i + 1 :] for i in range(len(word)))}


def _near(word: str, term: str) -> bool:
    if not term.endswith(GAP):
        return _one_typo(word, term)
    stem = term.removesuffix(GAP)
    return any(_one_typo(word[:length], stem) for length in (len(stem) - 1, len(stem), len(stem) + 1))


def _one_typo(word: str, term: str) -> bool:
    """At most one insertion, deletion, substitution or swap of neighbours for long terms; only a swap
    of neighbours for short ones."""
    if word == term:
        return True
    if len(term) < FUZZY_MIN_LETTERS:
        return len(word) == len(term) >= 3 and _one_swap(word, term)
    if abs(len(word) - len(term)) > 1:
        return False
    if len(word) == len(term):
        different = [i for i, (a, b) in enumerate(zip(word, term)) if a != b]
        return len(different) == 1 or _one_swap(word, term)
    shorter, longer = sorted((word, term), key=len)
    return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))


def _one_swap(word: str, term: str) -> bool:
    different = [i for i, (a, b) in enumerate(zip(word, term)) if a != b]
    return (
        len(different) == 2
        and different[1] == different[0] + 1
        and word[different[0]] == term[different[1]]
        and word[different[1]] == term[different[0]]
    )

"""T4 and the output guard's share of T6.

`python -m bench.sentences golden` scores vitae.sentences.split_all on the 52 English Golden Rules
from the pinned Pragmatic Segmenter README, cross-checked against pySBD's copy of the same cases.
`python -m bench.sentences time` times the Releaser per sentence on realistic answers.

Sentences are compared with whitespace runs collapsed and trimmed: the splitter keeps the text
verbatim (trailing whitespace stays with its sentence), while the golden answers are cleaned.
Rules 40 and 41 expect the errant newline to be dropped, which a verbatim splitter cannot do.
"""

import ast
import json
import re
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bench.fetch import CACHE, Entry, is_verified, load_manifest
from vitae.answer import OutputRules, Releaser
from vitae.generation import Delta, Finished
from vitae.sentences import split_all

README: Final = ("diasks2/pragmatic_segmenter", "README.md")
PYSBD: Final = ("nipunsadvilkar/pySBD", "tests/lang/test_english.py")
_RULE: Final = re.compile(r"^(\d+)\.\) \*\*(.+?)\*\*.*?\n```\n(.*?)\n=> (.*?)\n```", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class Case:
    rule: int
    title: str
    text: str
    expected: tuple[str, ...]


def pinned(repo: str, path: str, cache: Path = CACHE) -> Path:
    entry: Entry = next(e for e in load_manifest() if e.repo == repo and e.path == path)
    if not is_verified(entry, cache):
        raise SystemExit(f"{repo}/{path} is missing or fails its sha256. Run `python -m bench.fetch`.")
    return entry.local(cache)


def golden_rules(readme: str) -> tuple[Case, ...]:
    """The English section only. Inputs write a newline as \\n and some quotes as \\"; the expected
    lists are JSON, except rule 50, which lacks its closing bracket."""
    english = readme.split("#### Golden Rules (English)")[1].split("#### Golden Rules (German)")[0]
    cases: list[Case] = []
    for match in _RULE.finditer(english):
        number, title, text, expected = match.groups()
        text = text.replace("\\n", "\n").replace('\\"', '"')
        expected = expected.strip()
        if not expected.endswith("]"):
            expected += "]"
        cases.append(Case(int(number), title, text, tuple(json.loads(expected))))
    return tuple(cases)


def pysbd_cases(source: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """GOLDEN_EN_RULES_TEST_CASES, read with ast so nothing in the file runs. pytest.param entries
    (rule 18, marked xfail there) are unwrapped."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "GOLDEN_EN_RULES_TEST_CASES" for target in node.targets
        ):
            assert isinstance(node.value, ast.List)
            cases: list[tuple[str, tuple[str, ...]]] = []
            for element in node.value.elts:
                args = element.args if isinstance(element, ast.Call) else [element]
                pair = ast.literal_eval(args[0] if isinstance(element, ast.Call) else element)
                if isinstance(element, ast.Call):
                    pair = (pair, ast.literal_eval(args[1]))
                text, expected = pair
                cases.append((str(text), tuple(str(s) for s in expected)))
            return tuple(cases)
    raise ValueError("GOLDEN_EN_RULES_TEST_CASES not found")


def clean(sentences: Sequence[str]) -> tuple[str, ...]:
    return tuple(" ".join(s.split()) for s in sentences if s.strip())


def golden() -> int:
    cases = golden_rules(pinned(*README).read_text(encoding="utf-8"))
    misses = [
        (case, clean(split_all(case.text))) for case in cases if clean(split_all(case.text)) != clean(case.expected)
    ]
    passed = len(cases) - len(misses)
    print(f"Pragmatic Segmenter English Golden Rules: {passed}/{len(cases)} = {passed / len(cases):.1%}")
    for case, got in misses:
        print(f"  MISS rule {case.rule} ({case.title})")
        print(f"    expected {list(clean(case.expected))}")
        print(f"    got      {list(got)}")

    by_text = {case.text: case for case in cases}
    sbd = pysbd_cases(pinned(*PYSBD).read_text(encoding="utf-8"))
    unmatched = [text for text, _ in sbd if text not in by_text]
    disagree = [text for text, expected in sbd if text in by_text and clean(expected) != clean(by_text[text].expected)]
    sbd_pass = sum(clean(split_all(text)) == clean(expected) for text, expected in sbd)
    missing = sorted(case.rule for case in cases if case.text not in {text for text, _ in sbd})
    print(f"pySBD test_english.py: {len(sbd)} cases, {sbd_pass} pass ({sbd_pass / len(sbd):.1%})")
    print(f"  README rules absent from pySBD: {missing}")
    print(f"  pySBD inputs not in README: {len(unmatched)}; expectations that disagree: {len(disagree)}")
    for text in unmatched + disagree:
        print(f"    {text!r}")
    return 0 if passed / len(cases) >= 0.98 else 1


ANSWERS: Final = (
    "I'm a frontend developer at Example Studio, where I build design systems in React and TypeScript. "
    "Before that I spent three years at Sample Systems working on internal dashboards. "
    "Most of my day is spent pairing with designers and reviewing pull requests. "
    "I care a lot about accessibility, so every component ships with keyboard support and tests.",
    "Shift Notes is a small app for nurses to hand over patients between shifts. "
    "I built the frontend in React with Vitest for the tests, and a FastAPI backend in Python. "
    "The hardest part was making it work offline, because hospital Wi-Fi is unreliable. "
    "You can read more about it at https://alex.example/projects/shift-notes. "
    "It's used on two wards today, e.g. the night team uses it for every handover.",
    "My stack is mostly TypeScript, React and Node.js on the frontend. "
    "On the backend I reach for Python and FastAPI, with pytest for testing. "
    "I deploy with Docker and a plain CI pipeline, nothing fancy. "
    "If you want to reach me, email hello@alex.example and I'll get back to you within a day or two!",
    "I like working in small teams with short feedback loops. "
    "I write things down: design notes, decision records and clear pull request descriptions. "
    "When I disagree, I say so early and bring a prototype rather than an opinion. "
    "Outside work I hike, cook and play bass in a band that rehearses far too rarely.",
)


def _tokens(text: str, size: int = 4) -> list[str]:
    """Roughly what a streaming model sends: a few characters per delta."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _run(rules: OutputRules, deltas: Sequence[str]) -> int:
    releaser = Releaser(rules, ())
    for delta in deltas:
        releaser.feed(Delta(delta))
    releaser.feed(Finished("complete"))
    return releaser.sentences_released


def timing(repeats: int = 200, warmup: int = 50) -> int:
    rules = OutputRules.build(
        _example_prompt(),
        allowed_links=("https://alex.example",),
        allowed_contacts=("hello@alex.example",),
        allowed_names=("Alex Example",),
        refusal="I couldn't answer that one cleanly.",
    )
    streams = [_tokens(answer) for answer in ANSWERS]
    sentences = sum(_run(rules, stream) for stream in streams)
    assert sentences == sum(len(split_all(answer)) for answer in ANSWERS), "every sentence released"

    def batch() -> float:
        start = time.perf_counter_ns()
        for stream in streams:
            _run(rules, stream)
        return (time.perf_counter_ns() - start) / 1000 / sentences

    per_sentence = _measure(batch, repeats, warmup)
    quantiles = statistics.quantiles(per_sentence, n=100)
    print(
        f"Releaser, {len(ANSWERS)} answers, {sentences} sentences, 4-char deltas, {repeats} repeats after {warmup} warm-up"
    )
    print(
        f"  per sentence: p50 {statistics.median(per_sentence):.1f} us, p99 {quantiles[98]:.1f} us, "
        f"max {max(per_sentence):.1f} us (target 200 us)"
    )
    return 0 if quantiles[98] <= 200 else 1


def _measure(batch: Callable[[], float], repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        batch()
    return [batch() for _ in range(repeats)]


def _example_prompt() -> str:
    """A stand-in for prompt.system_prompt(profile) of the example profile, about the same length."""
    profile = Path(__file__).resolve().parents[1] / "profiles" / "example.toml"
    return profile.read_text(encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    match list(argv):
        case ["golden"]:
            return golden()
        case ["time"]:
            return timing()
        case _:
            print("usage: python -m bench.sentences golden|time", file=sys.stderr)
            return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""T6: what the pipeline costs outside the injection models. Run from backend/ as `python -m bench.overhead <command>`.

stages    normalize and screen in-process on real messages, as Engine.chat runs them, with the injection
          classifier stage replaced by a stub that returns PASS (the models are ADR 0001/0002's own
          budget). Per stage p50 and p99, the per-request total with the stub's cost subtracted, and
          each stage run on every message without the early exit
turns     the same messages as whole turns through engine.chat: everything but HTTP and the models, which
          is the per-request number T6 judges
app       a refused and an answered request through engine.chat and through the app on httpx.ASGITransport,
          scripted generator at zero delay, next to a bare FastAPI route on the same transport, split
          into stages, the engine's own cost, the app's and the transport's
release   OutputRules.check and the Releaser per sentence on realistic answers streamed in 4-character deltas
first     the sentence-buffer cost: first released text minus first token, with a generator that sends
          one token every 20 ms
all       stages, turns, app, release and first
digest    sha256 of every guard function's output over the dev texts, the recruiter questions and the
          Gretel dev documents; `--against` a previous digest file proves a speedup changed no output

Messages are every recruiter question, a per-corpus sample of the dev attacks the route accepts (at most
MAX_MESSAGE_CHARS) and every eval-case message.
Repeats are interleaved: each repeat runs every message once, in a fresh shuffled order, after warm-up.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from ipaddress import ip_address
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import patch

import httpx
import yaml
from fastapi import FastAPI, Response
from pydantic import ValidationError

from bench.corpora import RECRUITER, load_splits
from bench.fetch import CACHE, REPO_ROOT
from bench.pii import CORPORA as PII_CORPORA
from bench.pii import load as load_pii
from bench.sentences import ANSWERS as SENTENCE_ANSWERS
from vitae.answer import OutputRules, Releaser
from vitae.app import create_app
from vitae.contract import CHAT_PATH, ChatRequest
from vitae.engine import Engine, Reply
from vitae.generation import Delta, Finished, ScriptedGenerator
from vitae.guards import PASS, InputStages, Stage, Verdict
from vitae.guards.injection import Calibration
from vitae.guards.pii import find_pii
from vitae.observe import TurnRecord
from vitae.profile import load_profile
from vitae.screen import screen
from vitae.sentences import split_all
from vitae.settings import client_key, load_settings
from vitae.text import Turn, normalize

PROFILE: Final = load_profile(Path(__file__).resolve().parents[1] / "profiles" / "example.toml")
EVAL_CASES: Final = REPO_ROOT / "tests" / "evals" / "cases"
DIGEST: Final = CACHE / "bench" / "overhead-digest.json"
ORIGIN: Final = "https://alex.example"
PEER: Final = ("203.0.113.9", 52000)
ATTACKS_PER_CORPUS: Final = 150
TOKEN_CHARS: Final = 4
TOKEN_S: Final = 0.020
"""About what a small model on this CPU streams."""
REQUEST_BUDGET_MS: Final = 2.0
SENTENCE_BUDGET_US: Final = 200.0
REFUSED: Final = "Write me a sorting algorithm in Python"
ANSWERED: Final = "What projects have you built?"
AFTER_SCREEN: Final = ("queue", "prompt", "release")
"""TurnRecord.stage_ms entries the engine adds after screening."""

EXTRA_ANSWERS: Final = (
    "I joined Example Studio in March 2021 and led the design system rebuild in 2023. "
    "Before that I spent 2018 to 2021 at Sample Systems, where the dashboards served about 4,000 users. "
    "The rebuild cut bundle size by 38% and page load from 2.4 s to 1.1 s on a mid-range phone. "
    "Version 3.2 shipped in 6 weeks with 1,200 tests.",
    "Dear recruiter, thanks for asking about Neighbourhood Signals! "
    "It maps 311 reports for a city of 650,000 people, and I wrote it with Alex Example's usual stack. "
    "The API answers in under 50 ms at p95, and the code is at https://alex.example/projects/signals. "
    "Email hello@alex.example if you want a walkthrough.",
)
ANSWERS: Final = (*SENTENCE_ANSWERS, *EXTRA_ANSWERS)


class _ZeroClassifier:
    """Never called: stubbed() replaces the stage that would call it."""

    def score(self, text: str) -> float:
        return 0.0


def _pass(turn: Turn) -> Verdict:
    return PASS


CLASSIFIER_STUB: Final = Stage("injection_classifier", _pass)


def stubbed(stages: InputStages) -> InputStages:
    return InputStages(
        tuple(CLASSIFIER_STUB if stage.name == "injection_classifier" else stage for stage in stages.stages)
    )


class _Records:
    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []

    def turn(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def rejected(self, code: str) -> None:
        raise AssertionError(f"rejected {code}: the benchmark must not hit a limit")


SETTINGS: Final = load_settings({"VITAE_ALLOWED_ORIGINS": ORIGIN, "VITAE_RATE_LIMITS": "1000000/minute"})
PEER_KEY: Final = client_key(ip_address(PEER[0]))


def _engine(generator: ScriptedGenerator, observer: _Records) -> Engine:
    # Engine.build's own wiring, minus calibration.json: the stage that reads its threshold is stubbed,
    # and the file's shape moves with the cascade while this benchmark measures everything else.
    with patch.object(Calibration, "load", return_value=SimpleNamespace(injection_threshold=1.0)):
        engine = Engine.build(SETTINGS, PROFILE, generator=generator, classifier=_ZeroClassifier(), observer=observer)
    return replace(engine, stages=stubbed(engine.stages))


def tokens(text: str) -> list[str]:
    return [text[i : i + TOKEN_CHARS] for i in range(0, len(text), TOKEN_CHARS)]


def _p(values: Sequence[float], q: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if len(values) > 1 else values[0]


def _row(name: str, values: Sequence[float], unit: str = "ms", scale: float = 1.0) -> str:
    scaled = [value * scale for value in values]
    return (
        f"  {name:<22} p50 {statistics.median(scaled):8.3f} {unit}  p99 {_p(scaled, 99):8.3f} {unit}"
        f"  max {max(scaled):8.3f} {unit}  n={len(scaled)}"
    )


def recruiter_texts() -> list[str]:
    lines = RECRUITER.read_text(encoding="utf-8").splitlines()
    return [str(json.loads(line)["text"]) for line in lines if line.strip()]


def eval_messages() -> list[str]:
    messages: list[str] = []
    for path in sorted(EVAL_CASES.glob("*.yaml")):
        for case in cast(list[dict[str, object]], yaml.safe_load(path.read_text(encoding="utf-8"))):
            variables = cast(dict[str, object], case.get("vars", {}))
            messages += [str(turn) for turn in cast(list[object], variables.get("earlier_turns", []))]
            if "message" in variables:
                messages.append(str(variables["message"]))
    return messages


def accepted(text: str) -> bool:
    """Whether the route would screen it. A longer message is rejected as too_long before any guard runs."""
    try:
        ChatRequest(message=text)
    except ValidationError:
        return False
    return True


def attack_sample() -> list[str]:
    """Up to ATTACKS_PER_CORPUS dev attacks per corpus, among those the route accepts."""
    by_corpus: defaultdict[str, list[str]] = defaultdict(list)
    for sample in sorted(load_splits("injection").dev, key=lambda s: s.id):
        if sample.label == "attack" and accepted(sample.text):
            by_corpus[sample.corpus].append(sample.text)
    rng = random.Random(0)
    return [
        text
        for corpus in sorted(by_corpus)
        for text in rng.sample(by_corpus[corpus], min(ATTACKS_PER_CORPUS, len(by_corpus[corpus])))
    ]


@dataclass(frozen=True, slots=True)
class Messages:
    recruiter: list[str]
    attacks: list[str]
    evals: list[str]

    @classmethod
    def load(cls) -> "Messages":
        return cls(recruiter_texts(), attack_sample(), eval_messages())

    def all(self) -> list[str]:
        return [*self.recruiter, *self.attacks, *self.evals]


def _ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1e6


def run_stages(messages: Messages, repeats: int, warmup: int) -> None:
    engine = _engine(ScriptedGenerator.chunks(["."]), _Records())
    stages = engine.stages
    texts = messages.all()
    for _ in range(warmup):
        for text in texts:
            screen(normalize(text), stages)

    per_text: list[list[float]] = [[] for _ in texts]
    stage_ms: defaultdict[str, list[float]] = defaultdict(list)
    screen_bookkeeping: list[float] = []
    decided: Counter[str] = Counter()
    rng = random.Random(1)
    order = list(range(len(texts)))
    for repeat in range(repeats):
        rng.shuffle(order)
        for index in order:
            started = perf_counter_ns()
            turn = normalize(texts[index])
            normalized = perf_counter_ns()
            screening = screen(turn, stages)
            done = perf_counter_ns()
            timings = dict(screening.timings)
            stub = timings.get("injection_classifier", 0.0)
            per_text[index].append(_ms(started, done) - stub)
            stage_ms["normalize"].append(_ms(started, normalized))
            for name, ms in screening.timings:
                stage_ms[name].append(ms)
            screen_bookkeeping.append(_ms(normalized, done) - sum(timings.values()))
            if repeat == 0:
                decided[screening.decided_by or "proceed"] += 1

    totals = [ms for samples in per_text for ms in samples]
    medians = [statistics.median(samples) for samples in per_text]
    print(
        f"Stages, in-process: {len(messages.recruiter)} recruiter, {len(messages.attacks)} dev attacks, "
        f"{len(messages.evals)} eval messages; {repeats} interleaved repeats after {warmup} warm-up"
    )
    print(f"  stage calls {sum(len(v) for v in stage_ms.values())}, decided by (repeat 1) {dict(decided)}")
    for name, values in stage_ms.items():
        print(_row(name, values))
    print(_row("screen bookkeeping", screen_bookkeeping))
    print("Per request, normalize + screen, classifier stub subtracted:")
    print(_row("all samples", totals))
    print(_row("per-message median", medians))
    for label, group in (("recruiter", messages.recruiter), ("attacks", messages.attacks), ("evals", messages.evals)):
        start = texts.index(group[0])
        print(_row(f"  {label} medians", medians[start : start + len(group)]))

    turns = [normalize(text) for text in texts]
    print("Each stage on every message, no early exit (per-message median of 3):")
    for stage in stages.stages:
        if stage is CLASSIFIER_STUB:
            continue
        print(_row(stage.name, [_median_ns(stage.check, turn, 3) for turn in turns]))


async def _turns(messages: Messages, repeats: int, warmup: int) -> float:
    """Every message as a whole turn through engine.chat, drained: normalize, screen, the guard thread hop,
    the gate, the events, and for an answered turn the prompt and the Releaser over a five-sentence answer
    from a zero-delay generator. Returns the p99 in ms, classifier stub subtracted."""
    records = _Records()
    engine = _engine(ScriptedGenerator.chunks(tokens(ANSWERS[1])), records)
    texts = messages.all()
    for _ in range(warmup):
        for text in texts:
            await _through_engine(engine, text)
    totals: defaultdict[str, list[float]] = defaultdict(list)
    without_release: list[float] = []
    rng = random.Random(4)
    order = list(range(len(texts)))
    for _ in range(repeats):
        rng.shuffle(order)
        for index in order:
            records.turns.clear()
            elapsed, _ = await _through_engine(engine, texts[index])
            (record,) = records.turns
            timings = dict(record.stage_ms)
            totals[record.outcome].append(elapsed - timings.get("injection_classifier", 0.0))
            without_release.append(totals[record.outcome][-1] - timings.get("release", 0.0))
    every = [ms for values in totals.values() for ms in values]
    print(
        f"Per request through the engine, classifier stub subtracted, {repeats} interleaved repeats after {warmup} warm-up:"
    )
    for outcome, values in sorted(totals.items()):
        print(_row(outcome, values))
    print(_row("every turn", every))
    print(_row("minus the Releaser", without_release))
    p99 = _p(without_release, 99)
    print(
        f"  verdict: p99 {p99:.3f} ms without the Releaser against {REQUEST_BUDGET_MS} ms -> "
        f"{'PASS' if p99 <= REQUEST_BUDGET_MS else 'OVER'}. The Releaser has its own per-sentence budget (`release`), "
        "and the HTTP layer is on top (`app`)"
    )
    return p99


def _median_ns(check: Callable[[Turn], Verdict], turn: Turn, times: int) -> float:
    samples: list[float] = []
    for _ in range(times):
        started = perf_counter_ns()
        check(turn)
        samples.append(_ms(started, perf_counter_ns()))
    return statistics.median(samples)


async def _post(client: httpx.AsyncClient, message: str) -> tuple[float, bytes]:
    started = perf_counter_ns()
    response = await client.post(CHAT_PATH, json={"message": message}, headers={"Origin": ORIGIN})
    body = await response.aread()
    elapsed = _ms(started, perf_counter_ns())
    assert response.status_code == 200, (response.status_code, body)
    return elapsed, body


def _floor_app() -> FastAPI:
    app = FastAPI()
    body = b'event: session\ndata: {"session_id":"x","turns_left":1}\n\n'

    @app.post(CHAT_PATH)
    async def chat() -> Response:  # pyright: ignore[reportUnusedFunction]
        return Response(content=body, media_type="text/event-stream")

    return app


async def _through_engine(engine: Engine, message: str) -> tuple[float, list[str]]:
    started = perf_counter_ns()
    reply = await engine.chat(PEER_KEY, ChatRequest(message=message))
    assert isinstance(reply, Reply), reply
    kinds = [event.type async for event in reply]
    await reply.aclose()
    return _ms(started, perf_counter_ns()), kinds


async def _app_path(repeats: int, warmup: int) -> None:
    """The same two requests three ways, interleaved: engine.chat drained in-process, the app through
    httpx.ASGITransport, and a bare FastAPI route on that transport. The bare route is the load
    generator's own cost (httpx, the ASGI adapter, FastAPI routing), so it is subtracted."""
    records = _Records()
    engine = _engine(ScriptedGenerator.chunks(tokens(ANSWERS[1])), records)
    apps = {"floor": _floor_app(), "app": create_app(SETTINGS, engine, records)}
    clients = {
        name: httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=PEER), base_url="http://vitae.test")
        for name, app in apps.items()
    }
    cases = [("floor", "floor", ANSWERED)] + [
        (f"{path} {name}", path, message)
        for path in ("engine", "app")
        for name, message in (("refused", REFUSED), ("answered", ANSWERED))
    ]

    async def run(path: str, message: str) -> tuple[float, str]:
        if path == "engine":
            elapsed, kinds = await _through_engine(engine, message)
            return elapsed, ",".join(kinds)
        elapsed, body = await _post(clients[path], message)
        return elapsed, body.decode()

    for _ in range(warmup):
        for _, path, message in cases:
            await run(path, message)
            records.turns.clear()
    samples: defaultdict[str, list[float]] = defaultdict(list)
    engine_own: defaultdict[str, list[float]] = defaultdict(list)
    breakdown: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    outcomes: Counter[str] = Counter()
    rng = random.Random(2)
    for _ in range(repeats):
        rng.shuffle(cases)
        for name, path, message in cases:
            records.turns.clear()
            elapsed, seen = await run(path, message)
            samples[name].append(elapsed)
            if path == "floor":
                continue
            (record,) = records.turns
            outcomes[f"{name}:{record.outcome}"] += 1
            assert ("refused" in seen) == name.endswith("refused"), seen
            for stage, ms in record.stage_ms:
                breakdown[name][stage].append(ms)
            if path == "engine":
                engine_own[name].append(elapsed - sum(ms for _, ms in record.stage_ms))
    for client in clients.values():
        await client.aclose()

    print(f"App path, {repeats} interleaved repeats after {warmup} warm-up, outcomes {dict(outcomes)}")
    for name in sorted(samples):
        print(_row(name, samples[name]))
    print("Engine's own cost per request (engine.chat drained, minus the stage, queue, prompt and release timings):")
    for name in sorted(engine_own):
        print(_row(name, engine_own[name]))
    floor = statistics.median(samples["floor"])
    for name in ("refused", "answered"):
        stage_p50 = {stage: statistics.median(values) for stage, values in breakdown[f"engine {name}"].items()}
        engine_p50 = statistics.median(samples[f"engine {name}"])
        parts = {
            "screen stages": sum(ms for stage, ms in stage_p50.items() if stage not in AFTER_SCREEN),
            **{stage: ms for stage, ms in stage_p50.items() if stage in AFTER_SCREEN},
            "engine (normalize, thread hop, gate, events)": engine_p50 - sum(stage_p50.values()),
            "app (body, parse, SSE framing, response)": statistics.median(samples[f"app {name}"]) - floor - engine_p50,
            "transport floor": floor,
        }
        print(f"  {name} p50 split: " + ", ".join(f"{part} {ms:.3f}" for part, ms in parts.items()) + " ms")


def _released(rules: OutputRules, deltas: Sequence[str]) -> int:
    releaser = Releaser(rules, ())
    for delta in deltas:
        releaser.feed(Delta(delta))
    releaser.feed(Finished("complete"))
    return releaser.sentences_released


def run_release(repeats: int, warmup: int) -> float:
    rules = _engine(ScriptedGenerator.chunks(["."]), _Records()).output_rules
    sentences = [sentence for answer in ANSWERS for sentence in split_all(answer)]
    streams = [tokens(answer) for answer in ANSWERS]
    counts = [len(split_all(answer)) for answer in ANSWERS]
    assert [_released(rules, stream) for stream in streams] == counts, "every sentence released"
    for _ in range(warmup):
        for sentence in sentences:
            rules.check(sentence)
        for stream in streams:
            _released(rules, stream)

    check_us: list[float] = []
    releaser_us: list[float] = []
    rng = random.Random(3)
    order = list(range(len(ANSWERS)))
    for _ in range(repeats):
        rng.shuffle(order)
        for index in order:
            for sentence in split_all(ANSWERS[index]):
                started = perf_counter_ns()
                rules.check(sentence)
                check_us.append((perf_counter_ns() - started) / 1e3)
            started = perf_counter_ns()
            _released(rules, streams[index])
            releaser_us.append((perf_counter_ns() - started) / 1e3 / counts[index])
    print(
        f"Release: {len(ANSWERS)} answers, {len(sentences)} sentences, {TOKEN_CHARS}-char deltas, {repeats} interleaved repeats after {warmup} warm-up"
    )
    print(_row("OutputRules.check", check_us, "us"))
    print(_row("Releaser per sentence", releaser_us, "us"))
    p99 = _p(releaser_us, 99)
    print(
        f"  verdict: Releaser p99 {p99:.1f} us per sentence against {SENTENCE_BUDGET_US:.0f} us -> {'PASS' if p99 <= SENTENCE_BUDGET_US else 'OVER'}"
    )
    return p99


def _tokens_to_first_release(rules: OutputRules, deltas: Sequence[str]) -> int:
    """How many tokens after the first one the first sentence waits for."""
    releaser = Releaser(rules, ())
    for index, delta in enumerate(deltas):
        releaser.feed(Delta(delta))
        if releaser.sentences_released:
            return index
    raise AssertionError("no sentence released before the end")


async def _first_release(repeats: int) -> None:
    waits_ms: list[float] = []
    waits_tokens: list[int] = []
    first_sentence_tokens: list[int] = []
    for answer in ANSWERS:
        records = _Records()
        deltas = tokens(answer)
        engine = _engine(ScriptedGenerator.chunks(deltas, delay_s=TOKEN_S), records)
        waited = _tokens_to_first_release(engine.output_rules, deltas)
        first_sentence_tokens.append(-(-len(split_all(answer)[0]) // TOKEN_CHARS))
        for _ in range(repeats):
            reply = await engine.chat(PEER_KEY, ChatRequest(message=ANSWERED))
            assert isinstance(reply, Reply), reply
            async for event in reply:
                if event.type == "text":
                    break
            await reply.aclose()
            (record,) = records.turns
            records.turns.clear()
            assert record.first_delta_ms is not None and record.first_release_ms is not None, record
            waits_ms.append(record.first_release_ms - record.first_delta_ms)
            waits_tokens.append(waited)
    print(
        f"First release: {len(ANSWERS)} answers x {repeats}, one {TOKEN_CHARS}-char token every {TOKEN_S * 1000:.0f} ms"
    )
    print(_row("first text - first token", waits_ms))
    print(
        f"  in tokens after the first: median {statistics.median(waits_tokens):.0f}, range {min(waits_tokens)} to {max(waits_tokens)}; "
        f"first sentence length median {statistics.median(first_sentence_tokens):.0f} tokens, so the wait is the sentence "
        "itself plus the one token that confirms its end"
    )


def _texts_for_digest() -> tuple[list[str], list[str]]:
    """The dev texts plus every recruiter question, and the Gretel dev documents."""
    texts = list(dict.fromkeys([*(sample.text for sample in load_splits("injection").dev), *recruiter_texts()]))
    documents = [doc.text for corpus in PII_CORPORA for doc in load_pii(corpus, "dev").docs]
    return texts, documents


def _digest_items(name: str, values: Iterable[object]) -> dict[str, object]:
    items = [hashlib.sha256(repr(value).encode()).hexdigest()[:16] for value in values]
    combined = hashlib.sha256("".join(items).encode()).hexdigest()
    print(f"  {name:<14} {len(items):>7} outputs  sha256 {combined}")
    return {"count": len(items), "sha256": combined, "items": items}


def _release_events(rules: OutputRules, text: str) -> Iterator[object]:
    releaser = Releaser(rules, ())
    for delta in tokens(text):
        yield releaser.feed(Delta(delta))
    yield releaser.feed(Finished("complete"))


def digest(out: Path, against: Path | None) -> int:
    if os.environ.get("PYTHONHASHSEED") is None:
        # find_pii keeps the first of two overlapping name spans, in an order that comes from iterating a
        # set of strings, so its output depends on the hash seed. Fix the seed and compare like with like.
        sys.exit("digest needs a fixed hash seed: set PYTHONHASHSEED=0")
    texts, documents = _texts_for_digest()
    engine = _engine(ScriptedGenerator.chunks(["."]), _Records())
    stages = [stage for stage in engine.stages.stages if stage is not CLASSIFIER_STUB]
    rules = engine.output_rules
    print(f"Digest over {len(texts)} dev and recruiter texts and {len(documents)} Gretel dev documents")
    turns = [normalize(text) for text in texts]
    sentences = [sentence for text in [*texts, *documents] for sentence in split_all(text)]
    result = {
        "normalize": _digest_items("normalize", turns),
        "find_pii": _digest_items("find_pii", (find_pii(text) for text in [*texts, *documents])),
        **{stage.name: _digest_items(stage.name, (stage.check(turn) for turn in turns)) for stage in stages},
        "split_all": _digest_items("split_all", (split_all(text) for text in [*texts, *documents])),
        "output_check": _digest_items("output_check", (rules.check(sentence) for sentence in sentences)),
        "releaser": _digest_items("releaser", (tuple(_release_events(rules, text)) for text in [*texts, *documents])),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"  wrote {out}")
    if against is None:
        return 0
    previous = cast(dict[str, dict[str, object]], json.loads(against.read_text(encoding="utf-8")))
    failed = 0
    for name, now in result.items():
        before = previous[name]
        if before["sha256"] == now["sha256"]:
            print(f"  {name:<14} equal to {against.name} ({now['count']} outputs)")
            continue
        failed += 1
        pairs = zip(cast(list[str], before["items"]), cast(list[str], now["items"]), strict=False)
        differing = [index for index, (a, b) in enumerate(pairs) if a != b]
        print(
            f"  {name:<14} DIFFERS: {len(differing)} items, counts {before['count']} -> {now['count']}, first at {differing[:5]}"
        )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.overhead", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("stages", "turns", "app", "release", "first", "all", "digest"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DIGEST)
    parser.add_argument("--against", type=Path)
    args = parser.parse_args(argv)
    if args.command == "digest":
        return digest(args.out, args.against)
    over = False
    if args.command in ("stages", "all"):
        run_stages(Messages.load(), args.repeats, args.warmup)
    if args.command in ("turns", "all"):
        over |= asyncio.run(_turns(Messages.load(), args.repeats, args.warmup)) > REQUEST_BUDGET_MS
    if args.command in ("app", "all"):
        asyncio.run(_app_path(args.repeats * 25, args.warmup * 25))
    if args.command in ("release", "all"):
        over |= run_release(args.repeats * 10, args.warmup * 10) > SENTENCE_BUDGET_US
    if args.command in ("first", "all"):
        asyncio.run(_first_release(3))
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())

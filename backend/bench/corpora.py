"""Corpus rows for the guard benchmark, the split policy that decides which rows are held out, and
the two pools the injection benchmark scores.

Split policy:
- A corpus's official test split is held out; its train and validation splits are dev.
- A corpus without an official split is split by sha256 of the split-normalized text: a first hex
  digit 0 to 7 is dev, 8 to f is held out.
- Exact duplicates after split normalization are kept once, across all corpora. A text that appears
  on both sides stays held out only. A text with conflicting labels is dropped and reported.

Pools (the T1 pool definition was frozen before any held-out read):
- `t1_gate`: attacks recognisable from the message alone, against every benign corpus including
  the recruiter questions written for this project.
- `full`: every row of every fetched corpus.
"""

import ast
import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, cast

from bench.fetch import CACHE, MANIFEST, Entry, is_verified, load_manifest

Split = Literal["dev", "heldout"]
Label = Literal["attack", "benign"]
Detector = Literal["injection", "pii"]
Pool = Literal["t1_gate", "full"]
POOLS: Final[tuple[Pool, ...]] = ("t1_gate", "full")

RECRUITER: Final = Path(__file__).with_name("data") / "recruiter_benign.jsonl"

TRAINED_ON: Final[Mapping[str, tuple[str, ...]]] = {
    "gravitee-io/Llama-Prompt-Guard-2-22M-onnx": (),
    "protectai/deberta-v3-base-prompt-injection-v2": ("jackhhao",),
    "Horizon-Labs/prompt-injection-guard-small": ("deepset", "gandalf", "jackhhao"),
}
"""Corpora each model card names as training data. A model is never scored on them."""

csv.field_size_limit(1 << 30)


@dataclass(frozen=True, slots=True)
class Row:
    """One text as a corpus file gives it, before deduplication."""

    corpus: str
    text: str
    label: Label
    official: Split | None
    """The corpus's own split, if it has one."""
    t1_gate: bool
    """Counted by the T1 gate. False for attacks a message-only detector can't be asked to see."""


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    """sha256 of the split-normalized text. Stable across runs and machines."""
    text: str
    """The first occurrence's text, unnormalized: the guard sees what the corpus wrote."""
    label: Label
    corpus: str
    also_in: tuple[str, ...]
    t1_gate: bool
    """True when any of the merged rows counts for the T1 gate."""

    @property
    def corpora(self) -> tuple[str, ...]:
        return (self.corpus, *self.also_in)


@dataclass(frozen=True, slots=True)
class Splits:
    dev: tuple[Sample, ...]
    heldout: tuple[Sample, ...]
    conflicts: dict[str, int] = field(default_factory=dict[str, int])
    """Texts dropped for conflicting labels, counted per corpus."""

    def of(self, split: Split) -> tuple[Sample, ...]:
        return self.dev if split == "dev" else self.heldout


def split_key(text: str) -> str:
    """NFKC, whitespace runs collapsed, trimmed, casefolded. Deliberately not vitae.text.normalize:
    the guard's normalizer will change while it is tuned, and the held-out set must not move with it."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def digest(text: str) -> str:
    return hashlib.sha256(split_key(text).encode("utf-8")).hexdigest()


def hash_split(text: str) -> Split:
    return "dev" if int(digest(text)[0], 16) < 8 else "heldout"


@dataclass(slots=True)
class _Group:
    text: str
    labels: set[Label]
    corpora: list[str]
    splits: set[Split]
    t1_gate: bool


def split_rows(rows: Iterable[Row]) -> Splits:
    groups: dict[str, _Group] = {}
    for row in rows:
        if not row.text.strip():
            continue
        key = digest(row.text)
        group = groups.setdefault(key, _Group(row.text, set(), [], set(), False))
        group.labels.add(row.label)
        if row.corpus not in group.corpora:
            group.corpora.append(row.corpus)
        group.splits.add(row.official or hash_split(row.text))
        group.t1_gate = group.t1_gate or row.t1_gate

    dev: list[Sample] = []
    heldout: list[Sample] = []
    conflicts: dict[str, int] = {}
    for key in sorted(groups):
        group = groups[key]
        corpora = sorted(group.corpora)
        if len(group.labels) > 1:
            for corpus in corpora:
                conflicts[corpus] = conflicts.get(corpus, 0) + 1
            continue
        (label,) = group.labels
        sample = Sample(key, group.text, label, corpora[0], tuple(corpora[1:]), group.t1_gate)
        (heldout if "heldout" in group.splits else dev).append(sample)
    return Splits(tuple(dev), tuple(heldout), conflicts)


def in_pool(sample: Sample, pool: Pool) -> bool:
    return pool == "full" or sample.t1_gate


def eligible(samples: Iterable[Sample], trained_on: Iterable[str]) -> tuple[Sample, ...]:
    """samples minus every text that also appears in a corpus the model trained on."""
    excluded = frozenset(trained_on)
    return tuple(sample for sample in samples if excluded.isdisjoint(sample.corpora))


def open_verified(entry: Entry, cache: Path = CACHE) -> Path:
    """The cached file for entry. Raises if it is missing or its sha256 differs from the manifest."""
    if not is_verified(entry, cache):
        raise FileNotFoundError(
            f"{entry.local(cache)} is missing or fails its sha256 from the manifest. Run `python -m bench.fetch`."
        )
    return entry.local(cache)


class _Files:
    """The manifest's verified files for one repository, looked up by path pattern."""

    def __init__(self, entries: tuple[Entry, ...], repo: str, cache: Path) -> None:
        self._entries = [entry for entry in entries if entry.repo == repo]
        self._cache = cache

    def one(self, pattern: str) -> Path:
        (entry,) = [entry for entry in self._entries if re.fullmatch(pattern, entry.path)]
        return open_verified(entry, self._cache)

    def all(self, pattern: str) -> list[Path]:
        return [open_verified(entry, self._cache) for entry in self._entries if re.fullmatch(pattern, entry.path)]


def _parquet(path: Path) -> list[dict[str, object]]:
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    table = pq.read_table(path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast(list[dict[str, object]], table.to_pylist())  # pyright: ignore[reportUnknownMemberType]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


_TRAIN_TEST: Final[tuple[tuple[str, Split], ...]] = (("train", "dev"), ("test", "heldout"))
_JACKHHAO_LABELS: Final[dict[str, Label]] = {"jailbreak": "attack", "benign": "benign"}


def _deepset(files: _Files) -> Iterator[Row]:
    for split, official in _TRAIN_TEST:
        for record in _parquet(files.one(rf"data/{split}-.*\.parquet")):
            label: Label = "attack" if record["label"] == 1 else "benign"
            yield Row("deepset", str(record["text"]), label, official, True)


def _gandalf(files: _Files) -> Iterator[Row]:
    splits: tuple[tuple[str, Split], ...] = (("train", "dev"), ("validation", "dev"), ("test", "heldout"))
    for split, official in splits:
        for record in _parquet(files.one(rf"data/{split}-.*\.parquet")):
            yield Row("gandalf", str(record["text"]), "attack", official, True)


def _jackhhao(files: _Files) -> Iterator[Row]:
    for split, official in _TRAIN_TEST:
        for record in _csv(files.one(rf"default/jailbreak_dataset_{split}\.csv")):
            yield Row("jackhhao", record["prompt"], _JACKHHAO_LABELS[record["type"]], official, True)


def _spml(files: _Files) -> Iterator[Row]:
    for record in _csv(files.one(r"spml_prompt_injection\.csv")):
        if record["Prompt injection"] == "1":
            # Only rows built from a known attack seed are attacks without their system prompt.
            yield Row("spml", record["User Prompt"], "attack", None, record["Source"] == "Gandalf")
        else:
            yield Row("spml", record["User Prompt"], "benign", None, True)


def _jbb(files: _Files) -> Iterator[Row]:
    for record in _csv(files.one(r"data/benign-behaviors\.csv")):
        yield Row("jbb_benign", record["Goal"], "benign", None, True)
    for record in _csv(files.one(r"data/harmful-behaviors\.csv")):
        yield Row("jbb_harmful", record["Goal"], "attack", None, False)


_PROMPTINJECT_NAMES: Final = ("rogue_strings", "goal_hikacking_attacks", "prompt_leaking_attacks")


def _garak(files: _Files) -> Iterator[Row]:
    for path in files.all(r"garak/data/dan/.*\.json"):
        for text in cast(list[str], json.loads(path.read_bytes())):
            yield Row("garak_dan", text.replace("{generator.name}", "ChatGPT"), "attack", None, True)
    # Parsed, never imported: the downloaded file is data.
    tree = ast.parse(files.one(r"garak/resources/promptinject/prompt_data\.py").read_text(encoding="utf-8"))
    literals: dict[str, dict[str, object]] = {}
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(node, ast.Assign) and isinstance(target, ast.Name) and target.id in _PROMPTINJECT_NAMES:
            literals[target.id] = cast(dict[str, object], ast.literal_eval(node.value))
    rogues = [str(rogue) for rogue in literals["rogue_strings"].values()]
    for attack in literals["goal_hikacking_attacks"].values():
        instruction = str(cast(dict[str, object], attack)["instruction"])
        for rogue in rogues:
            yield Row("garak_promptinject", instruction.replace("{REPLACE_rogue_string}", rogue), "attack", None, True)
    for attack in literals["prompt_leaking_attacks"].values():
        yield Row("garak_promptinject", str(cast(dict[str, object], attack)["instruction"]), "attack", None, True)
    for path in files.all(r"garak/data/payloads/.*\.json"):
        for text in cast(dict[str, list[str]], json.loads(path.read_bytes()))["payloads"]:
            yield Row("garak_payloads", text, "attack", None, False)


def _recruiter(path: Path = RECRUITER) -> Iterator[Row]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield Row("recruiter", str(json.loads(line)["text"]), "benign", None, True)


def rows(detector: Detector, cache: Path = CACHE, manifest: Path = MANIFEST) -> Iterator[Row]:
    """Every row of every corpus the manifest lists for detector, each file read through open_verified."""
    entries = load_manifest(manifest)
    match detector:
        case "injection":
            yield from _deepset(_Files(entries, "deepset/prompt-injections", cache))
            yield from _gandalf(_Files(entries, "Lakera/gandalf_ignore_instructions", cache))
            yield from _jackhhao(_Files(entries, "jackhhao/jailbreak-classification", cache))
            yield from _spml(_Files(entries, "reshabhs/SPML_Chatbot_Prompt_Injection", cache))
            yield from _jbb(_Files(entries, "JailbreakBench/JBB-Behaviors", cache))
            yield from _garak(_Files(entries, "NVIDIA/garak", cache))
            yield from _recruiter()
        case "pii":
            raise NotImplementedError("PII rows carry spans, not one label; the T2 harness loads them")


def load_splits(detector: Detector, cache: Path = CACHE) -> Splits:
    return split_rows(rows(detector, cache))

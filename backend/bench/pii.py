"""The T2 harness: PII detection recall and precision per kind, and the input refusal rate on benign text.

`python -m bench.pii score --split dev` from backend/. `score --split heldout` refuses to run unless
the "pii" entry in bench/frozen.json matches the live files; `freeze` writes that entry. Each held-out
run is recorded in the committed bench/heldout-runs.jsonl, and while it holds a pii run another needs
--rerun, whatever bench/frozen.json now says.

Scoring:
- A gold span is detected when a find_pii hit of the same kind overlaps it (entity-level recall).
- A hit is correct when it overlaps a gold span of its own kind (precision). Unlabelled real PII in
  the corpora counts against precision, so precision is a floor.
- The refusal rate runs pii_stage on normalize(text), as the pipeline does, over rows with no gold
  span of a refused kind and no other identifier (OTHER_ID), plus the recruiter questions.
- Each corpus is reported on its own. Every rate carries a Wilson 95% interval.
"""

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, cast

from bench.corpora import RECRUITER, hash_split, open_verified
from bench.fetch import CACHE, load_manifest, source_sha256
from bench.guards import RUN_LOG, heldout_runs, log_heldout_run
from vitae.guards import Refuse
from vitae.guards.pii import REFUSED_AT_INPUT, PiiKind, find_pii, iban_ok, luhn_ok, pii_stage
from vitae.text import normalize

Split = Literal["dev", "heldout"]
Corpus = Literal["gretel_en", "gretel_finance"]
CORPORA: Final[tuple[Corpus, ...]] = ("gretel_en", "gretel_finance")

BACKEND: Final = Path(__file__).resolve().parents[1]
FROZEN: Final = Path(__file__).with_name("frozen.json")
FROZEN_FILES: Final = ("src/vitae/guards/pii.py", "src/vitae/text.py", "bench/pii.py")
"""What a held-out number depends on: the recognisers, the normalizer the stage reads, and this scorer."""

OTHER_ID: Final = "other_id"
NOT_PII: Final = "not_pii"
Use = PiiKind | Literal["other_id", "not_pii"]

LABELS: Final[Mapping[Corpus, Mapping[str, Use]]] = {
    "gretel_en": {
        "email": PiiKind.EMAIL,
        "phone_number": PiiKind.PHONE,
        "credit_card_number": PiiKind.CARD,
        "ssn": PiiKind.NATIONAL_ID,
        "national_id": PiiKind.NATIONAL_ID,
        "tax_id": PiiKind.NATIONAL_ID,
        "ipv4": PiiKind.IP,
        "ipv6": PiiKind.IP,
        "street_address": PiiKind.STREET_ADDRESS,
        "address": PiiKind.STREET_ADDRESS,
        "name": PiiKind.NAME,
        "first_name": PiiKind.NAME,
        "last_name": PiiKind.NAME,
        "medical_record_number": OTHER_ID,
        "customer_id": OTHER_ID,
        "employee_id": OTHER_ID,
        "license_plate": OTHER_ID,
        "user_name": OTHER_ID,
        "device_identifier": OTHER_ID,
        "bank_routing_number": OTHER_ID,
        "unique_identifier": OTHER_ID,
        "biometric_identifier": OTHER_ID,
        "account_number": OTHER_ID,
        "certificate_license_number": OTHER_ID,
        "vehicle_identifier": OTHER_ID,
        "api_key": OTHER_ID,
        "password": OTHER_ID,
        "health_plan_beneficiary_number": OTHER_ID,
        "swift_bic": OTHER_ID,
        "cvv": OTHER_ID,
        "pin": OTHER_ID,
        "coordinate": OTHER_ID,
        "url": OTHER_ID,
        "date_of_birth": NOT_PII,
        "date": NOT_PII,
        "date_time": NOT_PII,
        "time": NOT_PII,
        "company_name": NOT_PII,
        "city": NOT_PII,
        "postcode": NOT_PII,
        "country": NOT_PII,
        "state": NOT_PII,
    },
    "gretel_finance": {
        "email": PiiKind.EMAIL,
        "phone_number": PiiKind.PHONE,
        "credit_card_number": PiiKind.CARD,
        "iban": PiiKind.IBAN,
        "ssn": PiiKind.NATIONAL_ID,
        "ipv4": PiiKind.IP,
        "ipv6": PiiKind.IP,
        "street_address": PiiKind.STREET_ADDRESS,
        "name": PiiKind.NAME,
        "first_name": PiiKind.NAME,
        "last_name": PiiKind.NAME,
        "bban": OTHER_ID,
        "swift_bic_code": OTHER_ID,
        "customer_id": OTHER_ID,
        "employee_id": OTHER_ID,
        "bank_routing_number": OTHER_ID,
        "passport_number": OTHER_ID,
        "driver_license_number": OTHER_ID,
        "credit_card_security_code": OTHER_ID,
        "account_pin": OTHER_ID,
        "api_key": OTHER_ID,
        "user_name": OTHER_ID,
        "password": OTHER_ID,
        "local_latlng": OTHER_ID,
        "date": NOT_PII,
        "date_of_birth": NOT_PII,
        "date_time": NOT_PII,
        "time": NOT_PII,
        "company": NOT_PII,
    },
}
"""Every label each corpus uses. OTHER_ID labels are identifiers with no PiiKind (no format rule a
recogniser could check, or not in T2): not scored, and a row carrying one is not benign. NOT_PII
labels are not scored and leave a row benign. An unlisted label stops the run."""

_FILES: Final[Mapping[Corpus, Mapping[Split, tuple[str, ...]]]] = {
    "gretel_en": {
        "dev": ("data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"),
        "heldout": ("data/test-00000-of-00001.parquet",),
    },
    "gretel_finance": {
        "dev": ("data/train-00000-of-00001.parquet",),
        "heldout": ("data/test-00000-of-00001.parquet",),
    },
}
_REPOS: Final[Mapping[Corpus, str]] = {
    "gretel_en": "gretelai/gretel-pii-masking-en-v1",
    "gretel_finance": "gretelai/synthetic_pii_finance_multilingual",
}
_PLACEHOLDER: Final = re.compile(r"[\[\]{}<>]|\*{2,}|_{3,}|X{4,}")


@dataclass(frozen=True, slots=True)
class Gold:
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Doc:
    text: str
    gold: tuple[Gold, ...]


@dataclass(slots=True)
class Loaded:
    docs: list[Doc] = field(default_factory=list[Doc])
    unlocated: Counter[str] = field(default_factory=Counter[str])
    """Gold entities whose text does not occur in the document (Gretel en gives no offsets)."""
    placeholders: Counter[str] = field(default_factory=Counter[str])
    """Gold spans that are template leftovers such as "[Borrower Name]", dropped from scoring."""


def _parquet(path: Path, columns: list[str]) -> list[dict[str, object]]:
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    table = pq.read_table(path, columns=columns)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast(list[dict[str, object]], table.to_pylist())  # pyright: ignore[reportUnknownMemberType]


def _paths(corpus: Corpus, split: Split, cache: Path) -> list[Path]:
    entries = {(entry.repo, entry.path): entry for entry in load_manifest()}
    return [open_verified(entries[(_REPOS[corpus], path)], cache) for path in _FILES[corpus][split]]


def _keep(loaded: Loaded, label: str, text: str) -> bool:
    if not text.strip() or _PLACEHOLDER.search(text):
        loaded.placeholders[label] += 1
        return False
    return True


def load(corpus: Corpus, split: Split, cache: Path = CACHE, limit: int | None = None) -> Loaded:
    loaded = Loaded()
    labels = LABELS[corpus]
    for path in _paths(corpus, split, cache):
        if corpus == "gretel_en":
            for record in _parquet(path, ["text", "entities"])[:limit]:
                text = str(record["text"])
                gold: list[Gold] = []
                for entity in cast(list[dict[str, object]], ast.literal_eval(str(record["entities"]))):
                    value = str(entity["entity"]).strip()
                    for label in cast(list[str], entity["types"]):
                        if label not in labels:
                            raise ValueError(f"{corpus}: label {label!r} has no row in LABELS")
                        if not _keep(loaded, label, value):
                            continue
                        found = [m.start() for m in re.finditer(re.escape(value), text)]
                        if not found:
                            loaded.unlocated[label] += 1
                        gold.extend(Gold(label, start, start + len(value)) for start in found)
                loaded.docs.append(Doc(text, tuple(gold)))
        else:
            for record in _parquet(path, ["generated_text", "pii_spans"])[:limit]:
                text = str(record["generated_text"])
                gold = []
                for span in cast(list[dict[str, object]], json.loads(str(record["pii_spans"]))):
                    label, start, end = str(span["label"]), cast(int, span["start"]), cast(int, span["end"])
                    if label not in labels:
                        raise ValueError(f"{corpus}: label {label!r} has no row in LABELS")
                    if _keep(loaded, label, text[start:end]):
                        gold.append(Gold(label, start, end))
                loaded.docs.append(Doc(text, tuple(gold)))
    return loaded


def recruiter_texts(split: Split, path: Path = RECRUITER) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            text = str(json.loads(line)["text"])
            if hash_split(text) == split:
                texts.append(text)
    return texts


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True, slots=True)
class Rate:
    k: int
    n: int

    def render(self) -> str:
        if self.n == 0:
            return "n=0"
        low, high = wilson(self.k, self.n)
        return f"{100 * self.k / self.n:6.2f}% [{100 * low:.2f}, {100 * high:.2f}] ({self.k}/{self.n})"


def _checksum_valid(kind: PiiKind, value: str) -> bool | None:
    compact = re.sub(r"[ -]", "", value)
    match kind:
        case PiiKind.CARD:
            return compact.isdigit() and 13 <= len(compact) <= 19 and luhn_ok(compact)
        case PiiKind.IBAN:
            return iban_ok(compact)
        case _:
            return None


@dataclass(slots=True)
class Score:
    recall: dict[PiiKind, Rate]
    recall_valid: dict[PiiKind, Rate]
    """Card and IBAN gold whose own checksum holds. A number that fails Luhn or mod 97 is not one."""
    precision: dict[PiiKind, Rate]
    refusal: Rate
    refused_by: Counter[str]
    misses: dict[PiiKind, list[str]]
    miss_shapes: dict[PiiKind, Counter[str]]
    """Missed gold by shape (digits as 9, letters as A), to tell recogniser gaps from label noise."""
    false_hits: dict[PiiKind, list[str]]
    false_refusals: list[str]


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _refused(text: str) -> str | None:
    verdict = pii_stage().check(normalize(text))
    return verdict.rule if isinstance(verdict, Refuse) else None


def _shape(value: str) -> str:
    return re.sub(r"[^\W\d_]", "A", re.sub(r"\d", "9", value))[:40]


def _refused_spans(text: str) -> list[str]:
    seen = normalize(text).text
    return [seen[hit.start : hit.end] for hit in find_pii(seen) if hit.kind in REFUSED_AT_INPUT]


def score(corpus: Corpus, loaded: Loaded, examples: int = 5) -> Score:
    labels = LABELS[corpus]
    found: defaultdict[PiiKind, list[int]] = defaultdict(lambda: [0, 0])
    found_valid: defaultdict[PiiKind, list[int]] = defaultdict(lambda: [0, 0])
    correct: defaultdict[PiiKind, list[int]] = defaultdict(lambda: [0, 0])
    misses: defaultdict[PiiKind, list[str]] = defaultdict(list)
    miss_shapes: defaultdict[PiiKind, Counter[str]] = defaultdict(Counter)
    false_hits: defaultdict[PiiKind, list[str]] = defaultdict(list)
    refused, benign = 0, 0
    refused_by: Counter[str] = Counter()
    false_refusals: list[str] = []
    for doc in loaded.docs:
        hits = find_pii(doc.text)
        scored = [(gold, labels[gold.label]) for gold in doc.gold]
        for gold, use in scored:
            if not isinstance(use, PiiKind):
                continue
            hit = any(h.kind == use and _overlaps(h.start, h.end, gold.start, gold.end) for h in hits)
            found[use][0] += hit
            found[use][1] += 1
            valid = _checksum_valid(use, doc.text[gold.start : gold.end])
            if valid:
                found_valid[use][0] += hit
                found_valid[use][1] += 1
            if not hit:
                miss_shapes[use][_shape(doc.text[gold.start : gold.end])] += 1
            if not hit and valid is not False and len(misses[use]) < examples:
                context = doc.text[max(0, gold.start - 30) : gold.end + 30].replace("\n", " ")
                misses[use].append(f"{gold.label} {doc.text[gold.start : gold.end]!r} in ...{context}...")
        for h in hits:
            ok = any(use == h.kind and _overlaps(h.start, h.end, g.start, g.end) for g, use in scored)
            correct[h.kind][0] += ok
            correct[h.kind][1] += 1
            if not ok and len(false_hits[h.kind]) < examples:
                context = doc.text[max(0, h.start - 30) : h.end + 30].replace("\n", " ")
                false_hits[h.kind].append(f"{doc.text[h.start : h.end]!r} in ...{context}...")
        if all(use == NOT_PII or use == PiiKind.NAME for _, use in scored):
            benign += 1
            rule = _refused(doc.text)
            if rule:
                refused += 1
                refused_by[rule] += 1
                if len(false_refusals) < examples:
                    false_refusals.append(f"{rule}: {_refused_spans(doc.text)} in {doc.text[:100]!r}")
    return Score(
        recall={kind: Rate(k, n) for kind, (k, n) in found.items()},
        recall_valid={kind: Rate(k, n) for kind, (k, n) in found_valid.items()},
        precision={kind: Rate(k, n) for kind, (k, n) in correct.items()},
        refusal=Rate(refused, benign),
        refused_by=refused_by,
        misses=dict(misses),
        miss_shapes=dict(miss_shapes),
        false_hits=dict(false_hits),
        false_refusals=false_refusals,
    )


def score_recruiter(split: Split) -> tuple[Rate, list[str]]:
    texts = recruiter_texts(split)
    refusals = [f"{rule}: {text!r}" for text in texts if (rule := _refused(text))]
    return Rate(len(refusals), len(texts)), refusals


def render(corpus: Corpus, loaded: Loaded, result: Score) -> Iterator[str]:
    yield f"== {corpus}: {len(loaded.docs)} documents"
    yield f"{'kind':15} {'recall':40} {'recall, checksum-valid gold':40} precision"
    for kind in PiiKind:
        if kind in result.recall or kind in result.precision:
            recall = result.recall.get(kind, Rate(0, 0)).render()
            valid = result.recall_valid[kind].render() if kind in result.recall_valid else "-"
            precision = result.precision.get(kind, Rate(0, 0)).render()
            yield f"{kind.value:15} {recall:40} {valid:40} {precision}"
    yield f"input refusal on benign rows: {result.refusal.render()}  by rule {dict(result.refused_by)}"
    yield f"gold dropped as placeholders: {dict(loaded.placeholders)}"
    if loaded.unlocated:
        yield f"gold entities not found in their text: {dict(loaded.unlocated)}"
    for kind, shapes in result.miss_shapes.items():
        yield f"  missed {kind.value} by shape: {shapes.most_common(8)}"
    for kind, items in result.misses.items():
        yield f"  missed {kind.value} (checksum-invalid gold not shown): {items}"
    for kind, items in result.false_hits.items():
        yield f"  unmatched {kind.value} hits: {items}"
    for item in result.false_refusals:
        yield f"  refused benign row: {item}"


def frozen_files() -> dict[str, str]:
    return {path: source_sha256(BACKEND / path) for path in FROZEN_FILES}


def frozen_problems(frozen: Path = FROZEN) -> list[str]:
    """Why a held-out run may not start. Empty when bench/frozen.json pins the live files."""
    if not frozen.is_file():
        return [f"{frozen} does not exist. Run `python -m bench.pii freeze` once the dev split is final."]
    entry = json.loads(frozen.read_bytes()).get("pii")
    if not isinstance(entry, dict) or not isinstance(cast(dict[str, object], entry).get("files"), dict):
        return [f'{frozen} has no "pii" entry with "files".']
    pinned = cast(dict[str, str], cast(dict[str, object], entry)["files"])
    live = frozen_files()
    problems = [f"{path} is not pinned" for path in live if path not in pinned]
    problems += [f"{path} changed since freeze" for path, digest in live.items() if pinned.get(path, digest) != digest]
    return problems


def freeze(frozen: Path = FROZEN) -> str:
    """Writes the "pii" entry and keeps every other entry. Returns the entry's own sha256."""
    current = cast(dict[str, object], json.loads(frozen.read_bytes())) if frozen.is_file() else {}
    entry = {"files": frozen_files(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    current["pii"] = entry
    frozen.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(json.dumps(entry["files"], sort_keys=True).encode()).hexdigest()


def _frozen_digest() -> str:
    return hashlib.sha256(json.dumps(frozen_files(), sort_keys=True).encode()).hexdigest()


def run_score(split: Split, rerun: bool, examples: int, limit: int | None) -> int:
    if split == "heldout":
        problems = frozen_problems(FROZEN)
        if problems:
            print("refusing to score the held-out split:", *problems, sep="\n  ", file=sys.stderr)
            return 2
        if (previous := heldout_runs("pii", RUN_LOG)) and not rerun:
            print(
                f"the held-out split was already scored {len(previous)} time(s), first for frozen "
                f"{previous[0]['frozen']}; pass --rerun",
                file=sys.stderr,
            )
            return 2
    lines: list[str] = []
    summary: list[str] = []
    for corpus in CORPORA:
        loaded = load(corpus, split, limit=limit)
        result = score(corpus, loaded, examples)
        lines.extend(render(corpus, loaded, result))
        summary.append(f"{corpus} benign input refusal {result.refusal.render().strip()}")
    rate, refusals = score_recruiter(split)
    lines.append(f"== recruiter_benign ({split} by hash): input refusal {rate.render()}")
    lines.extend(f"  refused: {item}" for item in refusals)
    summary.append(f"recruiter input refusal {rate.render().strip()}")
    if split == "heldout":
        log_heldout_run("pii", _frozen_digest(), rerun=rerun, summary="; ".join(summary) + ".", run_log=RUN_LOG)
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.pii")
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--split", choices=("dev", "heldout"), required=True)
    scoring.add_argument("--rerun", action="store_true")
    scoring.add_argument("--examples", type=int, default=5)
    scoring.add_argument("--limit", type=int, help="score only the first N rows of each file, for quick iteration")
    commands.add_parser("freeze")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(f"froze pii: {freeze()}")
        return 0
    return run_score(
        cast(Split, args.split), cast(bool, args.rerun), cast(int, args.examples), cast(int | None, args.limit)
    )


if __name__ == "__main__":
    sys.exit(main())

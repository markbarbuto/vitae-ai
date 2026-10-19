"""The topic benchmark. Run from backend/ as `python -m bench.topics`.

Screens the dev half of the recruiter questions through the profile's three topic stages, in
pipeline order, and lists every row a stage refused or answered with availability. `--corpora`
also counts the scope stage's hits on the dev rows of the injection corpora, per label, corpus and rule.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from bench.corpora import RECRUITER, Row, load_splits, split_rows
from vitae.guards import Availability, InputStages, Refuse
from vitae.guards.topics import availability_stage, blocked_stage, scope_stage
from vitae.profile import Profile, load_profile
from vitae.screen import screen
from vitae.text import normalize

EXAMPLE_PROFILE: Final = Path(__file__).resolve().parents[1] / "profiles" / "example.toml"
TARGET_REFUSED: Final = 0.01
SHOWN_CHARS: Final = 110


def topic_stages(profile: Profile) -> InputStages:
    return InputStages(
        (
            blocked_stage(profile.topics),
            availability_stage(profile.availability),
            scope_stage(profile.topics, profile.assistant.addressing),
        )
    )


def recruiter_dev() -> list[tuple[str, str]]:
    """(kind, text) for each recruiter question on the dev side of bench.corpora's hash split."""
    kinds: dict[str, str] = {}
    rows: list[Row] = []
    for line in RECRUITER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            kinds.setdefault(record["text"], record["kind"])
            rows.append(Row("recruiter", record["text"], "benign", None, True))
    return [(kinds[sample.text], sample.text) for sample in split_rows(rows).dev]


def _shown(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SHOWN_CHARS else flat[: SHOWN_CHARS - 1] + "…"


def score_recruiter(stages: InputStages) -> list[str]:
    rows = recruiter_dev()
    hits: dict[str | None, list[str]] = {}
    for kind, text in rows:
        screening = screen(normalize(text), stages)
        match screening.decision:
            case Refuse(rule=rule) | Availability(rule=rule):
                hits.setdefault(screening.decided_by, []).append(f"  [{kind}] {rule}: {_shown(text)}")
            case _:
                pass
    scope = len(hits.get("scope", []))
    blocked = len(hits.get("blocked_topic", []))
    verdict = "PASS" if scope <= TARGET_REFUSED * len(rows) else "FAIL"
    lines = [
        f"recruiter dev rows: {len(rows)}",
        f"{verdict} refused by scope: {scope} ({scope / len(rows):.1%}), target <= {TARGET_REFUSED:.0%}",
        f"refused by blocked_topic: {blocked} ({blocked / len(rows):.1%})",
        f"availability answers: {len(hits.get('availability', []))}",
    ]
    for stage in ("scope", "blocked_topic", "availability"):
        if stage in hits:
            lines += [f"{stage}:", *hits[stage]]
    return lines


def score_corpora(stages: InputStages) -> list[str]:
    """How often the scope stage alone refuses a dev row of the injection corpora, per label and
    corpus. Informational: in the pipeline the injection stages run first and decide most attacks."""
    (scope,) = (stage for stage in stages.stages if stage.name == "scope")
    totals: Counter[tuple[str, str]] = Counter()
    hits: Counter[tuple[str, str, str]] = Counter()
    for sample in load_splits("injection").dev:
        totals[sample.label, sample.corpus] += 1
        verdict = scope.check(normalize(sample.text))
        if isinstance(verdict, Refuse):
            hits[sample.label, sample.corpus, verdict.rule] += 1
    lines: list[str] = []
    for label in ("attack", "benign"):
        lines.append(f"scope hits on dev {label} rows (scope stage alone):")
        for (row_label, corpus), total in sorted(totals.items()):
            if row_label != label:
                continue
            caught = {rule: n for (hit_label, name, rule), n in hits.items() if (hit_label, name) == (label, corpus)}
            count = sum(caught.values())
            detail = ", ".join(f"{rule} {n}" for rule, n in sorted(caught.items()))
            lines.append(f"  {corpus}: {count}/{total} ({count / total:.1%}){f' [{detail}]' if detail else ''}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.topics", description=__doc__)
    parser.add_argument("--profile", type=Path, default=EXAMPLE_PROFILE)
    parser.add_argument("--corpora", action="store_true", help="also count scope hits on the dev injection corpora")
    args = parser.parse_args(argv)
    stages = topic_stages(load_profile(args.profile))
    lines = score_recruiter(stages)
    if args.corpora:
        lines += score_corpora(stages)
    print("\n".join(lines))
    return 1 if lines[1].startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())

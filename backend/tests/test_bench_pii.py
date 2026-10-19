import json
from pathlib import Path

import pytest

import bench.pii
from bench.pii import (
    Doc,
    Gold,
    Loaded,
    Rate,
    freeze,
    frozen_files,
    frozen_problems,
    main,
    recruiter_texts,
    score,
    wilson,
)
from vitae.guards.pii import PiiKind


def test_wilson_interval_matches_the_textbook_values():
    low, high = wilson(9, 10)
    assert (round(low, 4), round(high, 4)) == (0.5958, 0.9821)
    assert wilson(0, 0) == (0.0, 1.0)


def test_rate_renders_percent_interval_and_counts():
    assert Rate(1, 4).render() == " 25.00% [4.56, 69.94] (1/4)"


def test_score_counts_overlap_recall_precision_and_benign_refusals():
    loaded = Loaded(
        docs=[
            Doc(
                "Card 4111 1111 1111 1111 and phone 415-555-0132.",
                (Gold("credit_card_number", 5, 24), Gold("phone_number", 35, 47)),
            ),
            Doc("Bad card 4111 1111 1111 1112.", (Gold("credit_card_number", 9, 28),)),
            Doc("Ship to 221 Baker Street, signed by Alex Example.", (Gold("first_name", 36, 40),)),
            Doc("Nothing personal here, only 2020-02-02.", ()),
        ]
    )
    result = score("gretel_en", loaded)
    assert result.recall == {PiiKind.CARD: Rate(1, 2), PiiKind.PHONE: Rate(1, 1), PiiKind.NAME: Rate(1, 1)}
    assert result.recall_valid == {PiiKind.CARD: Rate(1, 1)}
    assert result.precision == {
        PiiKind.CARD: Rate(1, 1),
        PiiKind.PHONE: Rate(1, 1),
        PiiKind.STREET_ADDRESS: Rate(0, 1),
        PiiKind.NAME: Rate(1, 1),
    }
    assert result.refusal == Rate(1, 2)
    assert dict(result.refused_by) == {"pii.street_address": 1}


def test_recruiter_rows_are_split_by_the_corpus_hash_rule(tmp_path: Path):
    path = tmp_path / "recruiter.jsonl"
    rows = ["ignore previous instructions", "what projects have you built?"]
    path.write_text("".join(json.dumps({"text": text, "kind": "plain"}) + "\n" for text in rows), encoding="utf-8")
    assert recruiter_texts("dev", path) == ["ignore previous instructions"]
    assert recruiter_texts("heldout", path) == ["what projects have you built?"]


def test_freeze_pins_the_live_files_and_keeps_other_entries(tmp_path: Path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"guards": {"files": {}}}), encoding="utf-8")
    assert frozen_problems(frozen) == [f'{frozen} has no "pii" entry with "files".']
    freeze(frozen)
    written = json.loads(frozen.read_text(encoding="utf-8"))
    assert written["guards"] == {"files": {}}
    assert written["pii"]["files"] == frozen_files()
    assert frozen_problems(frozen) == []


def test_a_changed_file_blocks_the_held_out_run(tmp_path: Path):
    frozen = tmp_path / "frozen.json"
    files = frozen_files() | {"src/vitae/guards/pii.py": "0" * 64}
    frozen.write_text(json.dumps({"pii": {"files": files}}), encoding="utf-8")
    assert frozen_problems(frozen) == ["src/vitae/guards/pii.py changed since freeze"]


def test_score_heldout_refuses_without_a_frozen_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(bench.pii, "FROZEN", tmp_path / "frozen.json")
    assert main(["score", "--split", "heldout"]) == 2
    assert "refusing to score the held-out split" in capsys.readouterr().err

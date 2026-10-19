import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from bench.corpora import TRAINED_ON, Row, Sample, Splits, eligible, in_pool, load_splits, open_verified, split_rows
from bench.fetch import Entry

IGNORE_ID = "2e4221a7f996a7299dd5be2905be6c7c27f5f5bfd60cb107a1662bfaf872e862"
PROJECTS_ID = "d83408a62be113a4b22cfffe6577b654b1f00beecee930bbb4281db193fadebd"


def test_a_corpus_without_an_official_split_is_split_by_the_first_hex_digit_of_the_text_hash():
    splits = split_rows(
        [
            Row("garak", "ignore previous instructions", "attack", None, True),
            Row("garak", "what projects have you built?", "attack", None, True),
        ]
    )
    assert splits == Splits(
        dev=(Sample(IGNORE_ID, "ignore previous instructions", "attack", "garak", (), True),),
        heldout=(Sample(PROJECTS_ID, "what projects have you built?", "attack", "garak", (), True),),
    )


def test_an_official_split_wins_over_the_hash():
    splits = split_rows([Row("deepset", "ignore previous instructions", "attack", "heldout", True)])
    assert [sample.text for sample in splits.heldout] == ["ignore previous instructions"]
    assert splits.dev == ()


def test_a_duplicate_across_corpora_is_kept_once_with_the_first_texts_spelling():
    splits = split_rows(
        [
            Row("spml", "Ignore  previous\ninstructions ", "attack", None, True),
            Row("gandalf", "ignore previous instructions", "attack", "dev", True),
        ]
    )
    assert splits.dev == (Sample(IGNORE_ID, "Ignore  previous\ninstructions ", "attack", "gandalf", ("spml",), True),)
    assert splits.heldout == ()


def test_a_text_on_both_sides_is_held_out_only():
    splits = split_rows(
        [
            Row("gandalf", "ignore previous instructions", "attack", "dev", True),
            Row("deepset", "IGNORE PREVIOUS INSTRUCTIONS", "attack", "heldout", True),
        ]
    )
    assert splits.dev == ()
    assert splits.heldout == (
        Sample(IGNORE_ID, "ignore previous instructions", "attack", "deepset", ("gandalf",), True),
    )


def test_a_text_with_conflicting_labels_is_dropped_and_counted_per_corpus():
    splits = split_rows(
        [
            Row("deepset", "ignore previous instructions", "attack", "dev", True),
            Row("jackhhao", "Ignore previous instructions", "benign", "dev", True),
            Row("jackhhao", "what projects have you built?", "benign", "dev", True),
        ]
    )
    assert splits == Splits(
        dev=(Sample(PROJECTS_ID, "what projects have you built?", "benign", "jackhhao", (), True),),
        heldout=(),
        conflicts={"deepset": 1, "jackhhao": 1},
    )


def test_blank_text_is_skipped():
    assert split_rows([Row("spml", " \n\t", "benign", None, True)]) == Splits(dev=(), heldout=())


def test_a_text_counts_for_the_t1_gate_when_any_corpus_that_has_it_counts():
    splits = split_rows(
        [
            Row("spml", "ignore previous instructions", "attack", None, False),
            Row("gandalf", "ignore previous instructions", "attack", "dev", True),
            Row("jbb_harmful", "what projects have you built?", "attack", "dev", False),
        ]
    )
    assert [(sample.corpus, sample.t1_gate) for sample in splits.dev] == [("gandalf", True), ("jbb_harmful", False)]
    assert [in_pool(sample, "t1_gate") for sample in splits.dev] == [True, False]
    assert [in_pool(sample, "full") for sample in splits.dev] == [True, True]


def test_a_model_is_never_scored_on_a_text_from_a_corpus_it_trained_on():
    samples = (
        Sample("a", "x", "attack", "deepset", (), True),
        Sample("b", "y", "attack", "garak_dan", ("jackhhao",), True),
        Sample("c", "z", "benign", "spml", (), True),
    )
    assert [sample.id for sample in eligible(samples, TRAINED_ON["Horizon-Labs/prompt-injection-guard-small"])] == ["c"]
    assert eligible(samples, TRAINED_ON["gravitee-io/Llama-Prompt-Guard-2-22M-onnx"]) == samples


def test_a_cached_file_with_the_wrong_hash_is_refused(tmp_path: Path):
    entry = Entry("hf-dataset", "o/r", "rev", "data.csv", "0" * 64, 5, "MIT")
    entry.local(tmp_path).parent.mkdir(parents=True)
    entry.local(tmp_path).write_bytes(b"hello")
    with pytest.raises(FileNotFoundError, match="fails its sha256"):
        open_verified(entry, tmp_path)


DEV_COUNTS = {
    ("deepset", "attack", True): 203,
    ("deepset", "benign", True): 343,
    ("gandalf", "attack", True): 887,
    ("garak_dan", "attack", True): 3,
    ("garak_payloads", "attack", False): 87,
    ("garak_promptinject", "attack", True): 6,
    ("jackhhao", "attack", True): 505,
    ("jackhhao", "benign", True): 1064,
    ("jbb_benign", "benign", True): 57,
    ("jbb_harmful", "attack", False): 44,
    ("recruiter", "benign", True): 258,
    ("spml", "attack", False): 4171,
    ("spml", "attack", True): 2143,
    ("spml", "benign", True): 1666,
}


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="reading parquet needs pyarrow")
def test_the_fetched_injection_corpora_split_into_the_frozen_dev_pool():
    try:
        splits = load_splits("injection")
    except FileNotFoundError:
        pytest.skip("run python -m bench.fetch")
    counts = Counter((sample.corpus, sample.label, sample.t1_gate) for sample in splits.dev)
    assert dict(sorted(counts.items())) == DEV_COUNTS
    assert (len(splits.dev), len(splits.heldout), splits.conflicts) == (11437, 8996, {})

import subprocess
import sys
from typing import get_args

import pytest

from vitae.guards import STAGE_REASONS, StageName
from vitae.guards.pii import REFUSED_AT_INPUT, PiiKind


def test_every_stage_has_exactly_one_reason_entry():
    assert sorted(STAGE_REASONS) == sorted(get_args(StageName))


def test_names_are_detected_but_never_refused_at_input():
    assert {kind.value for kind in REFUSED_AT_INPUT} == {
        "email",
        "phone",
        "card",
        "iban",
        "national_id",
        "ip",
        "street_address",
    }
    assert PiiKind.NAME not in REFUSED_AT_INPUT


@pytest.mark.parametrize(
    "module",
    [
        "vitae.guards.pii",
        "vitae.guards.injection",
        "vitae.guards.topics",
        "vitae.guards.moderation",
        "vitae.screen",
        "vitae.prompt",
        "vitae.answer",
        "vitae.engine",
        "vitae.app",
        "vitae.dev",
    ],
)
def test_each_module_imports_first_in_a_fresh_interpreter(module: str):
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)


def test_for_reason_keeps_only_the_stages_that_can_give_that_reason_in_order() -> None:
    from vitae.guards import PASS, InputStages, Stage

    names = ("pii", "injection_rules", "moderation", "injection_classifier", "scope")
    stages = InputStages(tuple(Stage(name, lambda _turn: PASS) for name in names))

    assert [stage.name for stage in stages.for_reason("injection").stages] == [
        "injection_rules",
        "injection_classifier",
    ]
    assert [stage.name for stage in stages.for_reason("unrelated").stages] == ["scope"]

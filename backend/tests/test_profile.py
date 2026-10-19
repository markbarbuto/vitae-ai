import tomllib
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from vitae.contract import RefusalReason
from vitae.profile import Profile

EXAMPLE = Path(__file__).parents[1] / "profiles" / "example.toml"


def example() -> dict[str, Any]:
    with EXAMPLE.open("rb") as file:
        return tomllib.load(file)


def test_the_example_profile_is_a_valid_profile_with_copy_for_every_refusal_reason():
    profile = Profile.model_validate(example())
    assert sorted(profile.refusals.root) == sorted(get_args(RefusalReason))


def test_a_profile_missing_copy_for_a_refusal_reason_is_refused():
    data = example()
    del data["refusals"]["output_rule"]
    with pytest.raises(ValidationError, match="missing copy for output_rule"):
        Profile.model_validate(data)


def test_a_profile_with_an_unknown_key_is_refused():
    data = example()
    data["assistant"]["persona"] = "pirate"
    with pytest.raises(ValidationError, match="assistant.persona"):
        Profile.model_validate(data)


def test_a_topic_id_that_is_not_a_slug_is_refused():
    data = example()
    data["topics"]["blocked"]["Home Life"] = ["kids"]
    with pytest.raises(ValidationError, match="topics.blocked.Home Life"):
        Profile.model_validate(data)

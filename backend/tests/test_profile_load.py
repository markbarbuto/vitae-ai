from pathlib import Path

import pytest

from vitae.contract import HostAction
from vitae.profile import ProfileError, compile_terms, load_profile

EXAMPLE = Path(__file__).parents[1] / "profiles" / "example.toml"


def test_the_example_profile_loads_with_its_copy_and_names():
    profile = load_profile(EXAMPLE)
    assert profile.refusals["pii"] == "Please don't share personal details here. Ask your question without them."
    assert profile.output.allowed_names == ("Alex Example",)
    assert profile.availability.action == HostAction(
        id="contact", label="Get in touch", href="mailto:hello@alex.example"
    )


def test_every_problem_in_a_profile_is_reported_at_once(tmp_path: Path):
    broken = EXAMPLE.read_text(encoding="utf-8")
    broken = broken.replace('home = ["home address"', 'Home = ["* address"')
    broken = broken.replace('"compan*"', '"pro*ject", "a*"')
    broken = broken.replace('output_rule = "I couldn\'t answer that one cleanly. Try asking another way."', "")
    broken = broken.replace('allowed_names = ["Alex Example"]', "")
    path = tmp_path / "profile.toml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ProfileError) as raised:
        load_profile(path)

    assert raised.value.problems == (
        'topics.allowed.experience.6: "pro*ject": * goes on its own or at the end of a word',
        'topics.allowed.experience.7: "a*": a word ending in * needs at least 3 letters before it',
        "topics.blocked.Home: String should match pattern '^[a-z][a-z0-9_]{0,31}$'",
        'topics.blocked.Home.0: "* address": a lone * stands for up to 4 words, so it goes between two words',
        "output.allowed_names: Field required",
        "refusals: missing copy for output_rule",
    )
    assert str(raised.value).startswith("Invalid profile (6 problems):\n  topics.allowed.experience.6")


def test_a_file_that_is_not_toml_is_one_problem(tmp_path: Path):
    path = tmp_path / "profile.toml"
    path.write_text("[assistant\nvoice = 1", encoding="utf-8")
    with pytest.raises(ProfileError) as raised:
        load_profile(path)
    assert len(raised.value.problems) == 1
    assert raised.value.problems[0].startswith(f"{path}: not valid TOML")


def test_a_missing_file_is_one_problem(tmp_path: Path):
    with pytest.raises(ProfileError) as raised:
        load_profile(tmp_path / "nope.toml")
    assert raised.value.problems[0].startswith(f"{tmp_path / 'nope.toml'}: cannot be read")


@pytest.mark.parametrize(
    ("term", "text", "found"),
    [
        ("open to * opportunities", "are you open to new opportunities?", True),
        ("open to * opportunities", "open to opportunities", True),
        ("open to * opportunities", "open to a b c d opportunities", True),
        ("open to * opportunities", "open to a b c d e opportunities", False),
        ("open to * opportunities", "open to new, exciting opportunities", True),
        ("project*", "my projects", True),
        ("project*", "my subprojects", False),
        ("pay you", "how much did they pay you?", True),
        ("pay you", "paying you", False),
        ("kids", "kidsmart toys", False),
        ("Example Studio", "at example  studio last year", True),
        ("Résumé", "send your resume", True),
        ("c++", "do you write c++?", True),
        ("a.b", "axb", False),
    ],
)
def test_terms_match_whole_words_with_star_gaps_and_prefixes(term: str, text: str, found: bool):
    assert (compile_terms([term]).search(text) is not None) is found


def test_no_terms_match_nothing():
    assert compile_terms([]).search("anything at all") is None

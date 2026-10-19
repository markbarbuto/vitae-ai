from pathlib import Path

from vitae.gate import Exchange
from vitae.profile import load_profile
from vitae.prompt import CANARY, ENGINE_INSTRUCTIONS, LEAK_MARKERS, build_messages, system_prompt
from vitae.screen import NO_EVIDENCE, Evidence
from vitae.text import normalize

PROFILE = load_profile(Path(__file__).parents[1] / "profiles" / "example.toml")


def test_the_canary_opens_the_engine_instructions_and_sits_again_just_before_the_hidden_rules():
    prompt = system_prompt(PROFILE)
    assert prompt.startswith(f"{CANARY}\nYou answer visitors' questions")
    assert prompt.count(CANARY) == 2
    assert prompt.endswith(f"{CANARY}\n\n{PROFILE.assistant.rules.strip()}")
    assert prompt.startswith(f"{ENGINE_INSTRUCTIONS}\n\nYou are the assistant on Alex Example's portfolio site.")
    assert "say so when it doesn't cover a question.\n\nVX7CANARY4B1E9D2A7C\n\nNever discuss family" in prompt


def test_the_leak_markers_are_the_canary_only():
    assert LEAK_MARKERS == ("VX7CANARY4B1E9D2A7C",)


def test_messages_are_system_then_history_pairs_then_the_normalized_turn():
    history = (Exchange("What projects have you built?", "I built Shift Notes. "),)
    turn = normalize("Tell me  more about\u200b it")
    evidence = Evidence(("Shift Notes is a handoff tool.", "It started in 2023."), ())

    messages = build_messages(PROFILE, history, turn, evidence)

    assert messages == [
        {
            "role": "system",
            "content": system_prompt(PROFILE)
            + "\n\nEvidence:\n\nShift Notes is a handoff tool.\n\nIt started in 2023.",
        },
        {"role": "user", "content": "What projects have you built?"},
        {"role": "assistant", "content": "I built Shift Notes. "},
        {"role": "user", "content": "Tell me more about it"},
    ]


def test_no_evidence_adds_no_evidence_section():
    messages = build_messages(PROFILE, (), normalize("Hi"), NO_EVIDENCE)
    assert messages == [{"role": "system", "content": system_prompt(PROFILE)}, {"role": "user", "content": "Hi"}]

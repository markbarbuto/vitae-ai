import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from vitae.contract import CHAT_PATH, DoneEvent, StartedEvent, TextEvent
from vitae.wire import read_events

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
MODELS = REPO / ".cache" / "models"
CASCADE = json.loads((BACKEND / "src" / "vitae" / "guards" / "calibration.json").read_bytes())["cascade"]
BASE_ENV = {
    "VITAE_ALLOWED_ORIGINS": "https://alex.example",
    "VITAE_RATE_LIMITS": "10/minute",
    "VITAE_PROFILE": str(BACKEND / "profiles" / "example.toml"),
}


def start_main(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    inherited = {name: value for name, value in os.environ.items() if not name.upper().startswith("VITAE_")}
    return subprocess.run(
        [sys.executable, "-c", "import vitae.main"],
        check=False,
        env={**inherited, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


def stderr_of(result: subprocess.CompletedProcess[str]) -> str:
    """stderr without onnxruntime's own log lines, which it prints on import on some Linux hosts."""
    return "".join(line for line in result.stderr.splitlines(keepends=True) if "onnxruntime" not in line)


def test_main_refuses_to_start_without_the_model_file(tmp_path: Path):
    result = start_main({**BASE_ENV, "VITAE_INJECTION_MODEL_DIR": str(tmp_path)})
    assert result.returncode == 1
    first = CASCADE[0]
    assert stderr_of(result).strip() == (
        f"No injection model at {tmp_path / first['dir'] / first['onnx']}. "
        "Set VITAE_INJECTION_MODEL_DIR or run `python -m bench.fetch`."
    )


def test_main_refuses_to_start_when_a_models_config_is_not_the_calibrated_one(tmp_path: Path):
    if not all((MODELS / model["dir"] / model["onnx"]).is_file() for model in CASCADE):
        pytest.skip("the calibrated injection models are not downloaded (python -m bench.fetch)")
    for model in CASCADE:
        for name in (model["onnx"], "config.json", "special_tokens_map.json", "tokenizer.json"):
            (tmp_path / model["dir"] / name).parent.mkdir(parents=True, exist_ok=True)
            os.link(MODELS / model["dir"] / name, tmp_path / model["dir"] / name)
    second = CASCADE[1]
    config = tmp_path / second["dir"] / "config.json"
    swapped = json.loads(config.read_bytes())
    swapped["id2label"] = {
        index: label for index, label in zip(swapped["id2label"], reversed(swapped["id2label"].values()), strict=True)
    }
    config.unlink()
    config.write_text(json.dumps(swapped), encoding="utf-8")

    result = start_main({**BASE_ENV, "VITAE_INJECTION_MODEL_DIR": str(tmp_path)})
    assert stderr_of(result).strip() == (
        f"{config} is not the file {BACKEND / 'src' / 'vitae' / 'guards' / 'calibration.json'} was calibrated on "
        "(sha256 d6264037c9aa4b3d821c1c9a0329572033adbdb622788ecca91c3636066d9622)."
    )
    assert result.returncode == 1


def test_main_refuses_to_start_on_bad_settings_and_lists_every_problem():
    result = start_main({"VITAE_RATE_LIMITS": "ten/minute"})
    assert result.returncode == 1
    assert stderr_of(result).splitlines()[:3] == [
        "Invalid configuration (2 problems):",
        "  VITAE_ALLOWED_ORIGINS: required.",
        '  VITAE_RATE_LIMITS: "ten/minute" is not count/unit with count >= 1 and unit second, minute, hour or day.',
    ]


def test_main_refuses_to_start_on_a_bad_profile(tmp_path: Path):
    profile = tmp_path / "profile.toml"
    profile.write_text("[assistant]\n", encoding="utf-8")
    result = start_main({**BASE_ENV, "VITAE_PROFILE": str(profile)})
    assert result.returncode == 1
    assert stderr_of(result).startswith("Invalid profile (")


def test_dev_serves_the_scripted_answer_through_the_real_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not all((MODELS / model["dir"] / model["onnx"]).is_file() for model in CASCADE):
        pytest.skip("the calibrated injection models are not downloaded (python -m bench.fetch)")
    for name, value in {**BASE_ENV, "VITAE_INJECTION_MODEL_DIR": str(MODELS)}.items():
        monkeypatch.setenv(name, value)
    script = tmp_path / "script.json"
    script.write_text('["I built Shift Notes.", " It helps support teams."]', encoding="utf-8")

    from vitae.dev import build

    app = build([str(script)])

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 4000))
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            return await client.post(
                CHAT_PATH, json={"message": "What projects have you built?"}, headers={"Origin": "https://alex.example"}
            )

    response = asyncio.run(scenario())
    assert response.status_code == 200
    assert list(read_events(response.text.splitlines()))[1:] == [
        StartedEvent(),
        TextEvent(text="I built Shift Notes. "),
        TextEvent(text="It helps support teams."),
        DoneEvent(finish="complete"),
    ]

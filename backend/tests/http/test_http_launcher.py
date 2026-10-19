import multiprocessing
import socket
import time
from pathlib import Path

import httpx
import pytest

from tests.http.harness import MODELS, ORIGIN, models_downloaded, run_dev, settings_env
from vitae.contract import CHAT_PATH


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_vitae_dev_keys_a_loopback_client_on_its_peer_whatever_x_forwarded_for_says(tmp_path: Path):
    if not models_downloaded():
        pytest.skip("vitae.dev needs the calibrated injection models (python -m bench.fetch)")
    script = tmp_path / "script.json"
    script.write_text('["I built Shift Notes."]', encoding="utf-8")
    port = free_port()
    env = settings_env(VITAE_RATE_LIMITS="2/minute", VITAE_INJECTION_MODEL_DIR=str(MODELS))
    argv = [str(script), "--port", str(port)]
    server = multiprocessing.get_context("spawn").Process(target=run_dev, args=(env, argv), daemon=True)
    server.start()
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                httpx.get(f"{base}/openapi.json", timeout=1)
                break
            except httpx.TransportError:
                assert server.is_alive() and time.monotonic() < deadline, "vitae.dev did not start"
                time.sleep(0.2)
        statuses = [
            httpx.post(
                f"{base}{CHAT_PATH}",
                json={"message": "What projects have you built?"},
                headers={"Origin": ORIGIN, "X-Forwarded-For": f"198.51.100.{n}"},
                timeout=30,
            ).status_code
            for n in range(3)
        ]
        assert statuses == [200, 200, 429]
    finally:
        server.terminate()
        server.join(10)

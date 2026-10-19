"""The app as vitae.dev wires it, served over real HTTP from a thread. Two test-only differences:
the scripted answer can be slowed down to hold slots, and `GET /_test/key` echoes the rate-limit key
the chat route derives for the same request. Production code never imports this module."""

import socket
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import FastAPI, Request

from vitae.app import create_app, peer_of
from vitae.engine import Engine
from vitae.generation import ScriptedGenerator
from vitae.guards.injection import Calibration, CascadeClassifier, InjectionClassifier
from vitae.observe import NULL_OBSERVER
from vitae.profile import load_profile
from vitae.settings import client_key, load_settings

BACKEND: Final = Path(__file__).resolve().parents[2]
REPO: Final = BACKEND.parent
MODELS: Final = REPO / ".cache" / "models"
ORIGIN: Final = "http://localhost:5173"
ANSWER: Final = ("I built Shift Notes. It helps", " support teams with", " handovers.")
KEY_PATH: Final = "/_test/key"


class _ZeroScore:
    def score(self, text: str) -> float:
        return 0.0


def models_downloaded() -> bool:
    return all((MODELS / model.dir / model.onnx).is_file() for model in Calibration.load().cascade)


@cache
def classifier() -> InjectionClassifier:
    """The calibrated models when they are downloaded, so these tests run the guards that ship. Without
    them (CI does not fetch models) a classifier that never flags, since nothing here tests injection."""
    if not models_downloaded():
        return _ZeroScore()
    return CascadeClassifier.load(MODELS, Calibration.load())


def settings_env(**overrides: str) -> dict[str, str]:
    return {
        "VITAE_ALLOWED_ORIGINS": ORIGIN,
        "VITAE_RATE_LIMITS": "1000/minute",
        "VITAE_PROFILE": str(BACKEND / "profiles" / "example.toml"),
        **overrides,
    }


def build_app(env: Mapping[str, str], *, delay_s: float = 0.0) -> FastAPI:
    settings = load_settings(env)
    engine = Engine.build(
        settings,
        load_profile(settings.profile_path),
        generator=ScriptedGenerator.chunks(ANSWER, delay_s=delay_s),
        classifier=classifier(),
        observer=NULL_OBSERVER,
    )
    app = create_app(settings, engine, NULL_OBSERVER)

    @app.get(KEY_PATH, include_in_schema=False)
    async def key(request: Request) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        peer = peer_of(request)
        return {"peer": peer, "key": str(client_key(settings.client_ip.resolve(peer, request.headers)))}

    return app


def run_dev(env: Mapping[str, str], argv: list[str]) -> None:
    """A multiprocessing target: vitae.dev's own launcher under env, so the process's pid is the server's."""
    import os

    from vitae.dev import main

    os.environ.update(env)
    main(argv)


@contextmanager
def serving(app: FastAPI, host: str = "127.0.0.1") -> Iterator[str]:
    """Yields the base URL of uvicorn serving app on a free port, and stops it on exit."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    # uvicorn's default proxy_headers=True rewrites the peer from X-Forwarded-For when it is 127.0.0.1,
    # before the app's own trusted-proxy check sees it. test_vitae_dev_launcher shows it on vitae.dev.
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off", proxy_headers=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()

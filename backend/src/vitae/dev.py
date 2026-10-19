"""The eval and HTTP-test server, `python -m vitae.dev <script.json>`. Same wiring as main.py, but the
generator replays a script. A separate entry point, never an env switch, so production cannot ship
a scripted generator by misconfiguration. The injection classifier is the real one, so the evals
test the guards that ship."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from vitae.app import create_app
from vitae.engine import Engine
from vitae.generation import ScriptedGenerator
from vitae.guards.injection import Calibration, CascadeClassifier
from vitae.observe import JsonLogObserver
from vitae.profile import load_profile
from vitae.settings import load_settings


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vitae.dev", description=__doc__)
    parser.add_argument("script", type=Path, help="JSON array of the chunks every answer replays")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args(argv)


def build(argv: Sequence[str]) -> FastAPI:
    """argv is the scripted generator's file, then optionally --host and --port."""
    arguments = _arguments(argv)
    settings = load_settings(os.environ)
    calibration = Calibration.load()
    if (problem := calibration.model_problem(settings.injection_model_dir)) is not None:
        sys.exit(problem)
    observer = JsonLogObserver(sys.stdout)
    engine = Engine.build(
        settings,
        load_profile(settings.profile_path),
        generator=ScriptedGenerator.from_file(arguments.script),
        classifier=CascadeClassifier.load(settings.injection_model_dir, calibration),
        observer=observer,
    )
    return create_app(settings, engine, observer)


def main(argv: Sequence[str]) -> None:
    arguments = _arguments(argv)
    uvicorn.run(build(argv), host=arguments.host, port=arguments.port, workers=1, proxy_headers=False)


if __name__ == "__main__":
    main(sys.argv[1:])

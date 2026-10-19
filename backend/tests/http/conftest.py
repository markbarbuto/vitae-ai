from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest

from tests.http.harness import build_app, serving, settings_env


@pytest.fixture
def serve() -> Iterator[Callable[..., str]]:
    """serve(delay_s=0.0, VITAE_X="...") starts the app with those VITAE_* overrides and returns its base URL."""
    with ExitStack() as stack:

        def start(delay_s: float = 0.0, **env: str) -> str:
            return stack.enter_context(serving(build_app(settings_env(**env), delay_s=delay_s)))

        yield start

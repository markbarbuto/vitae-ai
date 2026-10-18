"""promptfoo provider: sends each case to a running backend and reports what a visitor would see.

Runs on backend/.venv (npm run evals sets PROMPTFOO_PYTHON), so the outcome comes from the same
vitae.transcript fold the contract tests check against the committed example streams."""

import os
from collections.abc import Iterator
from time import perf_counter

import httpx

from vitae.contract import CHAT_PATH
from vitae.transcript import Transcript, transcript_of_response

BASE_URL = os.environ.get("VITAE_EVAL_BASE_URL", "http://localhost:8080")
ORIGIN = os.environ.get("VITAE_EVAL_ORIGIN", "http://localhost:5173")


def call_api(prompt: str, options: dict, context: dict) -> dict:
    metadata = context["test"].get("metadata") or {}
    origin = context.get("vars", {}).get("origin", ORIGIN)
    with httpx.Client(base_url=BASE_URL, timeout=120) as http:
        session_id, earlier = None, []
        for message in metadata.get("earlier_turns", []):
            turn = ask(http, message, session_id, ORIGIN, perf_counter())
            earlier.append(turn.outcome)
            session_id = turn.session_id or session_id
        started = perf_counter()
        turn = ask(http, prompt, session_id, origin, started)
    return {
        "output": turn.text,
        "latencyMs": round((perf_counter() - started) * 1000),
        "metadata": {
            "outcome": turn.outcome,
            "earlierOutcomes": earlier,
            "streamedThenFailed": turn.streamed_then_failed,
            "ttftMs": turn.first_text_ms,
            "promptMarkers": [],
        },
    }


def ask(http: httpx.Client, message: str, session_id: str | None, origin: str, started: float) -> Transcript:
    """An empty origin sends no Origin header at all."""
    headers = {"Origin": origin} if origin else {}
    body = {"message": message} | ({"session_id": session_id} if session_id else {})
    with http.stream("POST", CHAT_PATH, json=body, headers=headers) as response:
        return transcript_of_response(response.status_code, _until_dropped(response), perf_counter, started)


def _until_dropped(response: httpx.Response) -> Iterator[str]:
    """A connection lost mid-stream ends the lines, so fold grades the turn broken instead of the provider crashing."""
    try:
        yield from response.iter_lines()
    except httpx.TransportError:
        return

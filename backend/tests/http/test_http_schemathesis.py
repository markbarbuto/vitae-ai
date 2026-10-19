"""T8: Schemathesis over the app's own OpenAPI document, positive and negative bodies, with Origin
headers drawn alongside. Zero 5xx, and every non-200 is a contract ErrorResponse whose status is
REJECTION_STATUS of its code."""

from collections import Counter

import pytest
import schemathesis
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from schemathesis import Case, GenerationMode
from schemathesis.checks import (
    content_type_conformance,
    not_a_server_error,
    response_headers_conformance,
    response_schema_conformance,
    status_code_conformance,
)

from tests.http.harness import ORIGIN, build_app, settings_env
from vitae.contract import CHAT_PATH, REJECTION_STATUS, ErrorResponse

EXAMPLES_PER_MODE = 400
SUPPRESSED = [HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large]
REJECTION_CHECKS = [
    not_a_server_error,
    status_code_conformance,
    content_type_conformance,
    response_headers_conformance,
    response_schema_conformance,
]
STREAM_CHECKS = [not_a_server_error, status_code_conformance, content_type_conformance]
"""The 200 is an SSE stream: its body is not JSON, so only status and content type are checked."""
EXPECTED_STATUSES = {GenerationMode.POSITIVE: {200, 403, 429}, GenerationMode.NEGATIVE: {400, 403}}
ORIGINS = st.one_of(
    st.just(ORIGIN),
    st.just(ORIGIN),
    st.none(),
    st.sampled_from(
        [
            ORIGIN,
            ORIGIN.upper(),
            f"{ORIGIN}/",
            "http://localhost:5173.attacker.example",
            "https://attacker.example",
            "null",
            "",
            f"{ORIGIN}, https://attacker.example",
        ]
    ),
    # requests refuses to send a header value with leading whitespace, so none is drawn.
    st.text(st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=40).map(str.lstrip),
)


@pytest.mark.parametrize("mode", [GenerationMode.POSITIVE, GenerationMode.NEGATIVE], ids=lambda mode: mode.value)
def test_no_5xx_and_every_rejection_is_an_error_response_with_its_status(mode: GenerationMode):
    app = build_app(settings_env(VITAE_RATE_LIMITS="100/minute"))
    operation = schemathesis.openapi.from_asgi("/openapi.json", app)[CHAT_PATH]["POST"]
    statuses: Counter[int] = Counter()
    codes: Counter[str] = Counter()

    @given(case=operation.as_strategy(generation_mode=mode), origin=ORIGINS)
    @settings(max_examples=EXAMPLES_PER_MODE, deadline=None, suppress_health_check=SUPPRESSED, database=None)
    def one(case: Case, origin: str | None) -> None:
        response = case.call(headers={} if origin is None else {"Origin": origin})
        streamed = response.status_code == 200
        case.validate_response(response, checks=STREAM_CHECKS if streamed else REJECTION_CHECKS)
        statuses[response.status_code] += 1
        if response.status_code != 200:
            error = ErrorResponse.model_validate_json(response.content).error
            assert response.status_code == REJECTION_STATUS[error.code]
            codes[error.code] += 1

    one()
    print(f"{mode.value}: {sum(statuses.values())} cases, statuses {dict(statuses)}, codes {dict(codes)}")
    assert not [status for status in statuses if status >= 500]
    assert EXPECTED_STATUSES[mode] <= set(statuses)


def test_the_openapi_document_describes_the_200_as_an_event_stream():
    app = build_app(settings_env())
    content = app.openapi()["paths"][CHAT_PATH]["post"]["responses"]["200"]["content"]
    assert list(content) == ["text/event-stream"]

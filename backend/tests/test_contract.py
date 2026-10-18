import json

import pytest
from pydantic import ValidationError

from vitae.contract import ChatRequest, FinalRejection, HostAction, RetryableRejection, parse_chat_request
from vitae.wire import encode_rejection


def body(**fields: object) -> bytes:
    return json.dumps(fields).encode()


def test_a_500_character_message_is_accepted():
    assert parse_chat_request(body(message="a" * 500)) == ChatRequest(message="a" * 500)


def test_500_characters_are_counted_in_code_points_not_bytes():
    assert parse_chat_request(body(message="é" * 500)) == ChatRequest(message="é" * 500)


def test_a_501_character_message_is_too_long():
    assert parse_chat_request(body(message="a" * 501)) == FinalRejection(
        code="too_long", detail="message is over 500 characters."
    )


def test_a_body_over_8_kib_is_too_long_before_any_parsing():
    assert parse_chat_request(b"{" + b" " * 8192 + b"}") == FinalRejection(
        code="too_long", detail="Request body is over 8192 bytes."
    )


def test_an_unknown_field_is_an_invalid_request():
    assert parse_chat_request(body(message="hi", history=[])) == FinalRejection(
        code="invalid_request", detail="history: Extra inputs are not permitted"
    )


def test_a_blank_message_is_an_invalid_request():
    for blank in ("", "   ", "\n\t"):
        rejection = parse_chat_request(body(message=blank))
        assert isinstance(rejection, FinalRejection) and rejection.code == "invalid_request", blank


def test_malformed_json_is_an_invalid_request():
    rejection = parse_chat_request(b'{"message": ')
    assert isinstance(rejection, FinalRejection)
    assert rejection.code == "invalid_request"
    assert rejection.detail.startswith("body: Invalid JSON")


def test_a_session_id_is_kept_when_well_formed_and_refused_otherwise():
    assert parse_chat_request(body(message="hi", session_id="c2Vzc2lvbi1leGFtcGxlMQ")) == ChatRequest(
        message="hi", session_id="c2Vzc2lvbi1leGFtcGxlMQ"
    )
    assert parse_chat_request(body(message="hi", session_id="../../etc")) == FinalRejection(
        code="invalid_request", detail="session_id: String should match pattern '^[A-Za-z0-9_-]{22}$'"
    )


def test_a_retryable_rejection_is_a_status_a_json_body_and_retry_after():
    rejection = RetryableRejection(code="rate_limited", retry_after_s=42, detail="6 requests per minute per IP")
    assert encode_rejection(rejection) == (
        429,
        b'{"error":{"code":"rate_limited","retry_after_s":42,"detail":"6 requests per minute per IP"}}',
        {"Content-Type": "application/json", "Retry-After": "42"},
    )


def test_each_rejection_code_maps_to_its_status():
    statuses = {
        code: encode_rejection(FinalRejection(code=code, detail=""))[0]
        for code in ("invalid_request", "origin_not_allowed", "too_long", "session_limit")
    } | {
        code: encode_rejection(RetryableRejection(code=code, retry_after_s=1, detail=""))[0]
        for code in ("rate_limited", "busy", "unavailable")
    }
    assert statuses == {
        "invalid_request": 400,
        "origin_not_allowed": 403,
        "too_long": 413,
        "session_limit": 429,
        "rate_limited": 429,
        "busy": 503,
        "unavailable": 503,
    }


@pytest.mark.parametrize(
    "href",
    ["#contact", "/about", "https://owner.example/cv", "http://localhost:5173", "mailto:owner@example.com", "tel:+15550100"],
)
def test_an_action_may_link_to_a_page_a_mail_or_a_phone(href: str):
    assert HostAction(id="contact", label="Get in touch", href=href).href == href


@pytest.mark.parametrize(
    "href",
    ["javascript:alert(1)", "JavaScript:alert(1)", " javascript:alert(1)", "data:text/html,<script>", "vbscript:x", ""],
)
def test_an_action_cannot_carry_a_script_url(href: str):
    with pytest.raises(ValidationError):
        HostAction(id="contact", label="Get in touch", href=href)

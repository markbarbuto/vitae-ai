from ipaddress import ip_address, ip_network
from pathlib import Path

import pytest

from vitae.settings import (
    DirectPeer,
    Origin,
    RateLimit,
    Settings,
    SettingsError,
    TrustedProxy,
    client_key,
    load_settings,
    parse_origin,
)

MINIMAL = {"VITAE_ALLOWED_ORIGINS": "https://owner.example", "VITAE_RATE_LIMITS": "6/minute"}


def problems(env: dict[str, str]) -> tuple[str, ...]:
    with pytest.raises(SettingsError) as raised:
        load_settings(env)
    return raised.value.problems


def test_a_full_environment_loads_with_canonical_origins_and_defaults():
    env = {
        "VITAE_ALLOWED_ORIGINS": "HTTPS://Owner.Example:443/, http://localhost:5173",
        "VITAE_RATE_LIMITS": "6/minute,60/hour,200/day",
        "PATH": "/usr/bin",
    }
    assert load_settings(env) == Settings(
        allowed_origins=frozenset({Origin("https://owner.example"), Origin("http://localhost:5173")}),
        client_ip=DirectPeer(),
        rate_limits=(RateLimit(6, 60), RateLimit(60, 3600), RateLimit(200, 86400)),
        generation_slots=1,
        queue_size=4,
        session_max_turns=20,
        session_idle_s=1800,
        max_sessions=1000,
        profile_path=Path("profile.toml"),
        generation_url="http://localhost:8081/v1",
        generation_model="qwen3",
        injection_model_dir=Path(".cache/models"),
    )


def test_the_profile_generation_server_and_model_dir_can_be_set():
    env = {
        **MINIMAL,
        "VITAE_PROFILE": "profiles/example.toml",
        "VITAE_GENERATION_URL": "HTTP://Models.Internal:8081/v1/",
        "VITAE_GENERATION_MODEL": "qwen3-4b-instruct",
        "VITAE_INJECTION_MODEL_DIR": "/srv/models",
    }
    settings = load_settings(env)
    assert (
        settings.profile_path,
        settings.generation_url,
        settings.generation_model,
        settings.injection_model_dir,
    ) == (
        Path("profiles/example.toml"),
        "http://models.internal:8081/v1",
        "qwen3-4b-instruct",
        Path("/srv/models"),
    )


def test_a_generation_url_without_a_path_is_its_origin():
    assert load_settings({**MINIMAL, "VITAE_GENERATION_URL": "https://llm.example/"}).generation_url == (
        "https://llm.example"
    )


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        ("ftp://models.internal/v1", '"ftp://models.internal/v1" is not an http:// or https:// URL.'),
        ("localhost:8081/v1", '"localhost:8081/v1" is not an http:// or https:// URL.'),
        (
            "http://user:pw@models.internal/v1",
            '"http://user:pw@models.internal" has userinfo. An origin is scheme://host[:port] only.',
        ),
        (
            "http://models.internal/v1?key=1",
            '"http://models.internal/v1?key=1" has a query or fragment. '
            "Give the API base URL, such as http://localhost:8081/v1.",
        ),
    ],
)
def test_a_generation_url_that_is_not_an_http_base_url_is_refused(raw: str, problem: str):
    assert problems({**MINIMAL, "VITAE_GENERATION_URL": raw}) == (f"VITAE_GENERATION_URL: {problem}",)


def test_every_bad_value_is_reported_at_once():
    env = {
        "VITAE_ALLOWED_ORIGINS": "https://*.vercel.app,https://owner.example/chat",
        "VITAE_RATE_LIMIT": "5/minute",
        "VITAE_QUEUE_SIZE": "-1",
        "VITAE_SESSION_MAX_TURNS": "0",
    }
    assert problems(env) == (
        'VITAE_ALLOWED_ORIGINS: "https://*.vercel.app" is a wildcard. List each exact origin; '
        "anyone can deploy to a shared domain.",
        'VITAE_ALLOWED_ORIGINS: "https://owner.example/chat" has a path, query or fragment. '
        "An origin is scheme://host[:port] only.",
        "VITAE_RATE_LIMITS: required.",
        'VITAE_QUEUE_SIZE: "-1" is not a whole number >= 0.',
        'VITAE_SESSION_MAX_TURNS: "0" is not a whole number >= 1.',
        "VITAE_RATE_LIMIT: unknown variable. Did you mean VITAE_RATE_LIMITS?",
    )


def test_the_error_message_lists_every_problem():
    with pytest.raises(SettingsError) as raised:
        load_settings({"VITAE_RATE_LIMITS": "6/minute"})
    assert str(raised.value) == "Invalid configuration (1 problem):\n  VITAE_ALLOWED_ORIGINS: required."


@pytest.mark.parametrize(
    "raw",
    [
        "*",
        "https://*.vercel.app",
        "null",
        "https://owner.example/chat",
        "https://owner.example?x=1",
        "https://owner.example#top",
        "https://user@owner.example",
        "ftp://owner.example",
        "owner.example",
        "https://exämple.com",
        "https://owner.example:99999",
        "https://-owner.example",
        "http://[::1",
        "http://[example.com]",
    ],
)
def test_an_origin_that_is_not_one_exact_origin_is_refused(raw: str):
    with pytest.raises(ValueError):
        parse_origin(raw)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("https://owner.example", "https://owner.example"),
        ("https://owner.example/", "https://owner.example"),
        ("HTTPS://OWNER.EXAMPLE:443", "https://owner.example"),
        ("http://localhost:5173", "http://localhost:5173"),
        ("http://127.0.0.1:80", "http://127.0.0.1"),
        ("http://[::1]:8080", "http://[::1]:8080"),
        ("https://xn--exmple-cua.com", "https://xn--exmple-cua.com"),
    ],
)
def test_an_origin_is_stored_as_the_browser_sends_it(raw: str, canonical: str):
    assert parse_origin(raw) == canonical


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "null",
        "https://owner.example.evil.com",
        "https://evil-owner.example",
        "https://owner.example:8443",
        "http://owner.example",
        "HTTPS://OWNER.EXAMPLE",
        "https://owner.example/",
        " https://owner.example",
    ],
)
def test_only_the_exact_origin_is_allowed(header: str | None):
    settings = load_settings(MINIMAL)
    assert settings.origin_allowed("https://owner.example") is True
    assert settings.origin_allowed(header) is False


def test_a_malformed_url_is_reported_with_the_other_problems():
    env = {**MINIMAL, "VITAE_ALLOWED_ORIGINS": "http://[example.com],https://*.vercel.app"}
    assert problems(env) == (
        'VITAE_ALLOWED_ORIGINS: "http://[example.com]" is not a valid URL.',
        'VITAE_ALLOWED_ORIGINS: "https://*.vercel.app" is a wildcard. List each exact origin; '
        "anyone can deploy to a shared domain.",
    )


def test_an_unknown_vitae_variable_is_refused_in_any_case():
    env = {**MINIMAL, "VITAE_DEBUG": "1", "VITAE_ORIGINS": "https://x.example", "vitae_rate_limits": "1/second"}
    assert problems(env) == (
        "VITAE_DEBUG: unknown variable.",
        "VITAE_ORIGINS: unknown variable. Did you mean VITAE_ALLOWED_ORIGINS?",
        "vitae_rate_limits: unknown variable. Did you mean VITAE_RATE_LIMITS?",
    )


@pytest.mark.parametrize("header", ["X-Forwarded-For", "x-forwarded-for", "Forwarded"])
def test_a_list_valued_forwarding_header_is_refused(header: str):
    env = {**MINIMAL, "VITAE_TRUSTED_PROXY_CIDRS": "172.30.0.0/24", "VITAE_CLIENT_IP_HEADER": header}
    assert problems(env) == (
        f'VITAE_CLIENT_IP_HEADER: "{header}" is refused: proxies append to it, so its first entry is '
        "client-controlled. Have the proxy overwrite X-Real-IP instead.",
    )


@pytest.mark.parametrize("cidr", ["0.0.0.0/0", "::/0"])
def test_a_proxy_range_that_trusts_every_address_is_refused(cidr: str):
    assert problems({**MINIMAL, "VITAE_TRUSTED_PROXY_CIDRS": cidr}) == (
        f'VITAE_TRUSTED_PROXY_CIDRS: "{cidr}" trusts every address, so any client could set its own IP.',
    )


def test_a_proxy_range_with_host_bits_set_is_refused():
    assert problems({**MINIMAL, "VITAE_TRUSTED_PROXY_CIDRS": "172.30.0.5/24"}) == (
        'VITAE_TRUSTED_PROXY_CIDRS: "172.30.0.5/24" is not a CIDR block such as 172.30.0.0/24 '
        "(host bits must be zero).",
    )


def test_a_client_ip_header_without_proxy_ranges_is_refused():
    assert problems({**MINIMAL, "VITAE_CLIENT_IP_HEADER": "X-Real-IP"}) == (
        "VITAE_CLIENT_IP_HEADER: set without VITAE_TRUSTED_PROXY_CIDRS, so it would be ignored.",
    )


def test_a_repeated_rate_limit_window_is_refused():
    assert problems({**MINIMAL, "VITAE_RATE_LIMITS": "6/minute,10/minute"}) == (
        "VITAE_RATE_LIMITS: names a unit twice. Give each window once.",
    )


def test_proxy_ranges_select_trusted_proxy_mode_with_x_real_ip_by_default():
    settings = load_settings({**MINIMAL, "VITAE_TRUSTED_PROXY_CIDRS": "172.30.0.0/24, fd00::/8"})
    assert settings.client_ip == TrustedProxy(
        networks=(ip_network("172.30.0.0/24"), ip_network("fd00::/8")), header="X-Real-IP"
    )


SPOOFED = {"X-Real-IP": "198.51.100.7", "X-Forwarded-For": "198.51.100.7"}
PROXIED = {**MINIMAL, "VITAE_TRUSTED_PROXY_CIDRS": "172.30.0.0/24"}


def test_direct_mode_uses_the_peer_and_ignores_every_header():
    client_ip = load_settings(MINIMAL).client_ip
    assert client_ip.resolve(peer="203.0.113.9", headers=SPOOFED) == ip_address("203.0.113.9")


def test_a_trusted_proxy_peer_supplies_the_client_ip_through_its_header():
    client_ip = load_settings(PROXIED).client_ip
    assert client_ip.resolve(peer="172.30.0.2", headers={"x-real-ip": "198.51.100.7"}) == ip_address("198.51.100.7")


def test_an_untrusted_peer_cannot_set_its_ip_through_the_header():
    client_ip = load_settings(PROXIED).client_ip
    assert client_ip.resolve(peer="203.0.113.9", headers=SPOOFED) == ip_address("203.0.113.9")


def test_an_ipv4_peer_seen_through_a_dual_stack_socket_is_still_trusted():
    client_ip = load_settings(PROXIED).client_ip
    assert client_ip.resolve(peer="::ffff:172.30.0.2", headers=SPOOFED) == ip_address("198.51.100.7")


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Real-IP": "198.51.100.7, 10.0.0.1"}, {"X-Real-IP": "not an address"}, {"X-Real-IP": "fe80::1%eth0"}],
    ids=["missing", "list", "garbage", "zone id"],
)
def test_a_trusted_peer_without_one_clean_address_resolves_to_the_peer(headers: dict[str, str]):
    client_ip = load_settings(PROXIED).client_ip
    assert client_ip.resolve(peer="172.30.0.2", headers=headers) == ip_address("172.30.0.2")


def test_an_ipv4_client_the_proxy_writes_as_ipv6_maps_to_the_same_address():
    client_ip = load_settings(PROXIED).client_ip
    assert client_ip.resolve(peer="172.30.0.2", headers={"X-Real-IP": "::ffff:198.51.100.7"}) == ip_address(
        "198.51.100.7"
    )


def test_ipv6_addresses_in_one_64_share_a_rate_limit_key():
    rotated = [client_key(ip_address(f"2001:db8:1:2::{n:x}")) for n in (1, 0xBEEF, 0xFFFF)]
    assert rotated == [ip_network("2001:db8:1:2::/64")] * 3
    assert client_key(ip_address("2001:db8:1:3::1")) != rotated[0]


def test_ipv4_clients_keep_their_own_rate_limit_key():
    assert client_key(ip_address("198.51.100.7")) == ip_address("198.51.100.7")
    assert client_key(ip_address("198.51.100.8")) != client_key(ip_address("198.51.100.7"))


def test_an_ipv4_client_written_as_ipv6_is_keyed_as_ipv4():
    client_ip = load_settings(PROXIED).client_ip
    resolved = client_ip.resolve(peer="172.30.0.2", headers={"X-Real-IP": "::ffff:198.51.100.7"})
    assert client_key(resolved) == ip_address("198.51.100.7")

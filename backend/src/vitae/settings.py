"""Server configuration from VITAE_* environment variables. `python -m vitae.settings` checks the current environment."""

import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_address, ip_network
from typing import Final, NewType
from urllib.parse import urlsplit

Origin = NewType("Origin", str)
"""Browser serialization: scheme://host[:port], lowercase, no default port."""

IPAddress = IPv4Address | IPv6Address


class SettingsError(Exception):
    """Carries every problem at once, one line per problem, each starting with its variable."""

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        noun = "problem" if len(problems) == 1 else "problems"
        super().__init__("\n  ".join((f"Invalid configuration ({len(problems)} {noun}):", *problems)))


class _Invalid(ValueError):
    """One bad value, with a message that says how to fix it. load_settings prefixes the variable name."""


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_s: int


@dataclass(frozen=True)
class DirectPeer:
    """No proxy in front. The client IP is the TCP peer. Every forwarding header is ignored."""

    def resolve(self, peer: str, headers: Mapping[str, str]) -> IPAddress:
        return _peer_address(peer)


@dataclass(frozen=True)
class TrustedProxy:
    """A reverse proxy that overwrites `header` with the address it saw. The header is read only when
    the TCP peer is inside `networks`; any other peer is the client IP. A header that is missing,
    repeated, or holds anything but one address resolves to the peer."""

    networks: tuple[IPv4Network | IPv6Network, ...]
    header: str

    def resolve(self, peer: str, headers: Mapping[str, str]) -> IPAddress:
        address = _peer_address(peer)
        if not any(address in network for network in self.networks):
            return address
        values = [value for name, value in headers.items() if name.lower() == self.header.lower()]
        if len(values) != 1:
            return address
        try:
            forwarded = _canonical(ip_address(values[0].strip()))
        except ValueError:
            return address
        return address if isinstance(forwarded, IPv6Address) and forwarded.scope_id else forwarded


ClientIpSource = DirectPeer | TrustedProxy


def _peer_address(peer: str) -> IPAddress:
    return _canonical(ip_address(peer))


def _canonical(address: IPAddress) -> IPAddress:
    # Dual-stack sockets and some proxies write IPv4 clients as ::ffff:a.b.c.d. One client must map to one key.
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


ClientKey = IPv4Address | IPv6Network
IPV6_CLIENT_PREFIX: Final = 64


def client_key(address: IPAddress) -> ClientKey:
    """What per-IP limits count against. An IPv6 client usually holds a whole /64 and can pick a new
    address in it for every request, so IPv6 is keyed by its /64. IPv4 is keyed by the address."""
    if isinstance(address, IPv6Address):
        return IPv6Network((address, IPV6_CLIENT_PREFIX), strict=False)
    return address


@dataclass(frozen=True)
class Settings:
    allowed_origins: frozenset[Origin]
    client_ip: ClientIpSource
    rate_limits: tuple[RateLimit, ...]
    generation_slots: int
    queue_size: int
    """Waiting places beyond the slots. When full, requests get 503 busy."""
    session_max_turns: int
    session_idle_s: int
    max_sessions: int
    """At the cap, the least recently used session is evicted."""

    def origin_allowed(self, header: str | None) -> bool:
        """Exact string membership. The header is never parsed, so a lookalike can only fail."""
        return header is not None and header in self.allowed_origins


_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
_DEFAULT_PORTS: Final = {"http": 80, "https": 443}


def parse_origin(raw: str) -> Origin:
    """Accepts http/https, an ASCII host (punycode for IDN), an optional port, and at most a bare
    trailing "/". Rejects "*", any "*." pattern, "null", userinfo, paths, queries and fragments."""
    if not raw:
        raise _Invalid("has an empty entry.")
    if "*" in raw:
        raise _Invalid(f'"{raw}" is a wildcard. List each exact origin; anyone can deploy to a shared domain.')
    if raw.lower() == "null":
        raise _Invalid('"null" is the origin of sandboxed and file:// pages, which any site can produce.')
    try:
        parts = urlsplit(raw)
        host = parts.hostname or ""
    except ValueError:
        raise _Invalid(f'"{raw}" is not a valid URL.')
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not raw.lower().startswith(f"{scheme}://"):
        raise _Invalid(f'"{raw}" is not an http:// or https:// origin.')
    if "@" in parts.netloc:
        raise _Invalid(f'"{raw}" has userinfo. An origin is scheme://host[:port] only.')
    if parts.path not in ("", "/") or "?" in raw or "#" in raw:
        raise _Invalid(f'"{raw}" has a path, query or fragment. An origin is scheme://host[:port] only.')
    if not parts.netloc.isascii():
        raise _Invalid(f'"{raw}" has a non-ASCII host. Write it in punycode (xn--...), as browsers send it.')
    if not _valid_host(host):
        raise _Invalid(f'"{raw}" has no valid host.')
    try:
        port = parts.port
    except ValueError:
        raise _Invalid(f'"{raw}" has an invalid port.')
    if port == 0:
        raise _Invalid(f'"{raw}" has an invalid port.')
    shown_host = f"[{host}]" if ":" in host else host
    shown_port = "" if port in (None, _DEFAULT_PORTS[scheme]) else f":{port}"
    return Origin(f"{scheme}://{shown_host}{shown_port}")


def _valid_host(host: str) -> bool:
    try:
        ip_address(host)
        return True
    except ValueError:
        return bool(host) and all(_LABEL.match(label) for label in host.split("."))


def _each[T](parse_one: Callable[[str], T]) -> Callable[[str], tuple[T, ...]]:
    """A comma-separated list parser that reports every bad entry, not just the first."""

    def parse(raw: str) -> tuple[T, ...]:
        problems, values = [], []
        for entry in (part.strip() for part in raw.split(",")):
            try:
                values.append(parse_one(entry))
            except _Invalid as problem:
                problems.append(str(problem))
        if problems:
            raise _Invalid("\n".join(problems))
        return tuple(values)

    return parse


_UNIT_SECONDS: Final = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
_DIGITS = re.compile(r"[0-9]+")


def _rate_limit(entry: str) -> RateLimit:
    count, _, unit = entry.partition("/")
    if not _DIGITS.fullmatch(count) or int(count) < 1 or unit not in _UNIT_SECONDS:
        raise _Invalid(f'"{entry}" is not count/unit with count >= 1 and unit second, minute, hour or day.')
    return RateLimit(requests=int(count), window_s=_UNIT_SECONDS[unit])


def _rate_limits(raw: str) -> tuple[RateLimit, ...]:
    limits = _each(_rate_limit)(raw)
    windows = [limit.window_s for limit in limits]
    if len(set(windows)) != len(windows):
        raise _Invalid("names a unit twice. Give each window once.")
    return limits


def _network(entry: str) -> IPv4Network | IPv6Network:
    try:
        network = ip_network(entry, strict=True)
    except ValueError:
        raise _Invalid(f'"{entry}" is not a CIDR block such as 172.30.0.0/24 (host bits must be zero).')
    if network.prefixlen == 0:
        raise _Invalid(f'"{entry}" trusts every address, so any client could set its own IP.')
    return network


def _networks(raw: str) -> tuple[IPv4Network | IPv6Network, ...]:
    return _each(_network)(raw) if raw else ()


_REFUSED_HEADERS: Final = frozenset({"x-forwarded-for", "forwarded"})
_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]+$")


def _ip_header(raw: str) -> str:
    if raw.lower() in _REFUSED_HEADERS:
        raise _Invalid(
            f'"{raw}" is refused: proxies append to it, so its first entry is client-controlled. '
            "Have the proxy overwrite X-Real-IP instead."
        )
    if not _HEADER_NAME.match(raw):
        raise _Invalid(f'"{raw}" is not a header name.')
    return raw


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        if not _DIGITS.fullmatch(raw) or int(raw) < minimum:
            raise _Invalid(f'"{raw}" is not a whole number >= {minimum}.')
        return int(raw)

    return parse


@dataclass(frozen=True)
class _Var[T]:
    parse: Callable[[str], T]
    default: str | None
    """None means required. A default is parsed like any value, so it cannot drift from the rules."""


_VARS: Final[Mapping[str, _Var]] = {
    "VITAE_ALLOWED_ORIGINS": _Var(_each(parse_origin), None),
    "VITAE_TRUSTED_PROXY_CIDRS": _Var(_networks, ""),
    "VITAE_CLIENT_IP_HEADER": _Var(_ip_header, "X-Real-IP"),
    "VITAE_RATE_LIMITS": _Var(_rate_limits, None),
    "VITAE_GENERATION_SLOTS": _Var(_int_at_least(1), "1"),
    "VITAE_QUEUE_SIZE": _Var(_int_at_least(0), "4"),
    "VITAE_SESSION_MAX_TURNS": _Var(_int_at_least(1), "20"),
    "VITAE_SESSION_IDLE_MINUTES": _Var(_int_at_least(1), "30"),
    "VITAE_MAX_SESSIONS": _Var(_int_at_least(1), "1000"),
}


def load_settings(env: Mapping[str, str]) -> Settings:
    """Pure: reads only the given mapping. Collects every problem, then raises SettingsError. An
    empty value counts as unset. Any VITAE_* name it does not know is a problem too, so a typo
    cannot silently fall back to a default."""
    problems: list[str] = []
    values = {}
    for name, var in _VARS.items():
        raw = env.get(name, "").strip()
        if not raw and var.default is None:
            problems.append(f"{name}: required.")
            continue
        try:
            values[name] = var.parse(raw or var.default)
        except _Invalid as problem:
            problems.extend(f"{name}: {line}" for line in str(problem).splitlines())

    if env.get("VITAE_CLIENT_IP_HEADER", "").strip() and not env.get("VITAE_TRUSTED_PROXY_CIDRS", "").strip():
        problems.append("VITAE_CLIENT_IP_HEADER: set without VITAE_TRUSTED_PROXY_CIDRS, so it would be ignored.")

    for name in sorted(env):
        if name.upper().startswith("VITAE_") and name not in _VARS:
            guess = get_close_matches(name.upper(), _VARS, n=1)
            hint = f" Did you mean {guess[0]}?" if guess else ""
            problems.append(f"{name}: unknown variable.{hint}")

    if problems:
        raise SettingsError(tuple(problems))
    networks = values["VITAE_TRUSTED_PROXY_CIDRS"]
    return Settings(
        allowed_origins=frozenset(values["VITAE_ALLOWED_ORIGINS"]),
        client_ip=TrustedProxy(networks, values["VITAE_CLIENT_IP_HEADER"]) if networks else DirectPeer(),
        rate_limits=values["VITAE_RATE_LIMITS"],
        generation_slots=values["VITAE_GENERATION_SLOTS"],
        queue_size=values["VITAE_QUEUE_SIZE"],
        session_max_turns=values["VITAE_SESSION_MAX_TURNS"],
        session_idle_s=values["VITAE_SESSION_IDLE_MINUTES"] * 60,
        max_sessions=values["VITAE_MAX_SESSIONS"],
    )


def describe(settings: Settings) -> str:
    match settings.client_ip:
        case DirectPeer():
            client_ip = "TCP peer (no proxy; forwarding headers ignored)"
        case TrustedProxy(networks=networks, header=header):
            client_ip = f"{header} from peers in {', '.join(map(str, networks))}"
    limits = ", ".join(f"{limit.requests} per {limit.window_s} s" for limit in settings.rate_limits)
    return "\n".join(
        (
            f"Allowed origins:  {', '.join(sorted(settings.allowed_origins))}",
            f"Client IP:        {client_ip}",
            f"Rate limits:      {limits} per IP",
            f"Generation:       {settings.generation_slots} slots, {settings.queue_size} queue places",
            f"Sessions:         {settings.session_max_turns} turns, idle after {settings.session_idle_s} s, "
            f"at most {settings.max_sessions}",
        )
    )


if __name__ == "__main__":
    try:
        print(describe(load_settings(os.environ)))
    except SettingsError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

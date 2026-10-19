"""Forwarding-header sets for the nginx differential: hand-picked shapes, then seeded random ones.
Each set is a list of (name, value) pairs sent verbatim, duplicates and odd casing included."""

import random
from typing import Final

TRUSTED_V4: Final = "172.31.250.5"
TRUSTED_V6: Final = "fd00:7a17::5"
UNTRUSTED_V4: Final = "172.31.250.100"
UNTRUSTED_V6: Final = "fd00:7a17::100"
NGINX_V4: Final = "172.31.250.2"
NGINX_V6: Final = "fd00:7a17::2"

Headers = list[tuple[str, str]]

NAMES: Final = (
    "X-Forwarded-For",
    "x-forwarded-for",
    "X-FORWARDED-FOR",
    "X-Real-IP",
    "x-real-ip",
    "X-Real-Ip",
    "Forwarded",
    "X-Forwarded",
    "X-Client-IP",
    "True-Client-IP",
    "CF-Connecting-IP",
    "X-Originating-IP",
    "X_Forwarded_For",
    "X_Real_IP",
)

FIXED_VALUES: Final = (
    "",
    " ",
    "unknown",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "::",
    "255.255.255.255",
    TRUSTED_V4,
    TRUSTED_V6,
    NGINX_V4,
    NGINX_V6,
    UNTRUSTED_V4,
    "::ffff:203.0.113.9",
    "::ffff:127.0.0.1",
    f"::ffff:{TRUSTED_V4}",
    "::ffff:cb00:7109",
    "0:0:0:0:0:ffff:203.0.113.9",
    "2001:db8::1",
    "2001:DB8::1",
    "2001:0db8:0000:0000:0000:0000:0000:0001",
    "fe80::1%eth0",
    "2001:db8::1/64",
    "203.0.113.9/32",
    "203.0.113.9:443",
    "[2001:db8::1]:443",
    "[2001:db8::1]",
    "999.1.1.1",
    "1.2.3",
    "01.02.03.04",
    "0x7f.0.0.1",
    "2130706433",
    "203.0.113.9 ",
    "\t203.0.113.9",
    '"203.0.113.9"',
    "203.0.113.9, 198.51.100.7",
    "1.1.1.1;2.2.2.2",
    "_hidden",
    "localhost",
    "\x80\xff",
    "a" * 2000,
)


def _random_value(rng: random.Random) -> str:
    match rng.randrange(8):
        case 0:
            return ".".join(str(rng.randrange(256)) for _ in range(4))
        case 1:
            return ":".join(f"{rng.randrange(65536):x}" for _ in range(8))
        case 2:
            return f"2001:db8:{rng.randrange(65536):x}::{rng.randrange(65536):x}"
        case 3:
            return f"::ffff:{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(256)}"
        case 4:
            return rng.choice((TRUSTED_V4, TRUSTED_V6, NGINX_V4, UNTRUSTED_V6))
        case 5:
            return "".join(rng.choice('0123456789abcdef:.[]%, ;="') for _ in range(rng.randrange(1, 30)))
        case _:
            return rng.choice(FIXED_VALUES)


def _chain(rng: random.Random) -> str:
    separator = rng.choice((", ", ",", " , ", ",,"))
    entries = [_random_value(rng) for _ in range(rng.randrange(1, 6))]
    if rng.random() < 0.4:
        entries.append(rng.choice((TRUSTED_V4, TRUSTED_V6)))
    return separator.join(entries)


def _forwarded(rng: random.Random) -> str:
    def element() -> str:
        value = _random_value(rng)
        quoted = f'"[{value}]:{rng.randrange(65536)}"' if ":" in value else value
        return f"for={rng.choice((value, quoted, '_hidden', 'unknown'))};proto=https"

    return ", ".join(element() for _ in range(rng.randrange(1, 4)))


def _random_set(rng: random.Random) -> Headers:
    headers: Headers = []
    for _ in range(rng.randrange(1, 5)):
        name = rng.choice(NAMES)
        value = _forwarded(rng) if name == "Forwarded" else _chain(rng)
        headers.append((name, value))
    return headers


def header_sets(random_count: int = 520, seed: int = 7) -> list[Headers]:
    fixed: list[Headers] = [[]]
    fixed += [[(name, value)] for name in ("X-Forwarded-For", "X-Real-IP", "Forwarded") for value in FIXED_VALUES]
    fixed += [
        [("X-Forwarded-For", "203.0.113.9"), ("X-Forwarded-For", TRUSTED_V4)],
        [("X-Forwarded-For", TRUSTED_V4), ("X-Forwarded-For", "203.0.113.9")],
        [("X-Real-IP", "203.0.113.9"), ("X-Real-IP", "198.51.100.7")],
        [("X-Real-IP", "203.0.113.9"), ("x-real-ip", "198.51.100.7")],
        [("X-Forwarded-For", f"203.0.113.9, {TRUSTED_V4}"), ("X-Real-IP", "198.51.100.7")],
        [("X-Forwarded-For", f"2001:db8::9, {TRUSTED_V6}, {TRUSTED_V4}")],
        [("X-Forwarded-For", ", ".join(f"10.0.{n // 256}.{n % 256}" for n in range(200)))],
        [("Forwarded", 'for="[2001:db8::1]:443";proto=https'), ("X-Forwarded-For", "2001:db8::1")],
    ]
    rng = random.Random(seed)
    return fixed + [_random_set(rng) for _ in range(random_count)]

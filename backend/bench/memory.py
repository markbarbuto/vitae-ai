"""T5 probe: memory the gate holds after a flood of distinct client keys, and with full sessions.

`python -m bench.memory` from backend/. Each scenario runs in its own process twice: once under
tracemalloc for Python heap growth, once without it for the RSS delta, because tracemalloc's own
bookkeeping would swamp RSS.
"""

import ctypes
import json
import os
import subprocess
import sys
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Network
from pathlib import Path
from typing import Final

from vitae.gate import (
    BUCKETS_PER_WINDOW,
    HISTORY_EXCHANGES,
    MAX_TRACKED_CLIENTS,
    Exchange,
    Gate,
    GateLimits,
    Generate,
    ScreenPermit,
    Stop,
)
from vitae.settings import ClientKey, RateLimit

BUDGET_BYTES: Final = 10_000_000
FLOOD_KEYS: Final = 1_000_000
FLOOD_RPS: Final = 1000
ONE_PER_WINDOW: Final = (RateLimit(1, 60), RateLimit(1, 3600), RateLimit(1, 86400))
"""One request per minute, hour and day. Any count up to 256 costs the same, since CPython shares
those int objects; N=1 lets the probe prove every key is still tracked."""
SATURATING: Final = (RateLimit(10, 60), RateLimit(20, 3600), RateLimit(30, 86400))
BIG_COUNTS: Final = (RateLimit(2000, 60), RateLimit(4000, 3600), RateLimit(6000, 86400))
BIG_COUNT_REPEAT: Final = 257
"""Hits per scheduled time, so every bucket holds a count CPython does not share."""
BIG_COUNT_SAMPLE: Final = 512
MAX_SESSIONS: Final = 1000
MESSAGE_CHARS: Final = 500
ANSWER_CHARS: Final = 2000
"""400 generation tokens at about 5 characters each."""


@dataclass(frozen=True, slots=True)
class Growth:
    scenario: str
    held: int
    """Keys or sessions the gate holds at the end."""
    growth_bytes: int
    peak_bytes: int

    @property
    def per_item_bytes(self) -> float:
        return self.growth_bytes / self.held


def gate_limits(rate: tuple[RateLimit, ...]) -> GateLimits:
    return GateLimits(
        rate_limits=rate, slots=1, queue_size=4, session_max_turns=20, session_idle_s=1800, max_sessions=MAX_SESSIONS
    )


def mixed_keys(stop: int, start: int = 0) -> Iterator[ClientKey]:
    """Alternates IPv4 addresses and IPv6 /64s, all distinct."""
    for n in range(start, stop):
        if n % 2:
            yield IPv6Network(((0x20010DB8 << 96) | (n << 64), 64))
        else:
            yield IPv4Address(n)


def admitted(gate: Gate, key: ClientKey, now: float) -> bool:
    result = gate.throttle(key, None, now)
    if isinstance(result, ScreenPermit):
        result.release()
        return True
    return False


def measured(run: Callable[[], tuple[Gate, int]], scenario: str) -> Growth:
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    gate, held = run()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del gate
    return Growth(scenario=scenario, held=held, growth_bytes=current - base, peak_bytes=peak - base)


def run_flood(keys: int) -> tuple[Gate, int]:
    """Every key once, FLOOD_RPS a second, all inside the longest window, so nothing is swept and
    the cap alone bounds memory. Then every key the cap should keep is hit again: each one is
    refused (still counted), and the newest key past the cap is admitted (dropped)."""
    gate = Gate(gate_limits(ONE_PER_WINDOW))
    for n, key in enumerate(mixed_keys(keys)):
        assert admitted(gate, key, n / FLOOD_RPS)
    end = keys / FLOOD_RPS
    kept = min(keys, MAX_TRACKED_CLIENTS)
    held = sum(1 for key in mixed_keys(keys, start=keys - kept) if not admitted(gate, key, end))
    assert held == kept, f"{kept - held} keys the cap should keep were dropped"
    for dropped in mixed_keys(keys - kept, start=max(0, keys - kept - 1)):
        assert admitted(gate, dropped, end), "a key past the cap is still tracked"
    return gate, held


def flood(keys: int) -> Growth:
    return measured(lambda: run_flood(keys), f"flood {keys:,} keys")


def schedule() -> tuple[list[float], float]:
    """Hit times that leave every bucket of every SATURATING window non-zero at the last one: one hit
    in each of the 6 older day buckets, the 6 older hour buckets and all 7 minute buckets."""
    minute, hour, day = (limit.window_s // BUCKETS_PER_WINDOW for limit in SATURATING)
    end = 7 * day + 13000
    end_minute, end_hour, end_day = end // minute, end // hour, end // day
    times = sorted(
        [float(b * day + 1) for b in range(end_day - 6, end_day)]
        + [float(b * hour + 1) for b in range(end_hour - 6, end_hour)]
        + [float(b * minute + 1) for b in range(end_minute - 6, end_minute + 1)]
    )
    for limit in SATURATING:
        width = limit.window_s / BUCKETS_PER_WINDOW
        last = int(times[-1] // width)
        assert {int(t // width) for t in times if t // width > last - 7} == set(range(last - 6, last + 1))
    return times, times[-1]


def run_saturated(keys: int, rate: tuple[RateLimit, ...], repeat: int) -> tuple[Gate, int]:
    gate = Gate(gate_limits(rate))
    clients = list(mixed_keys(keys))
    times, _ = schedule()
    for now in times:
        for key in clients:
            for _ in range(repeat):
                assert admitted(gate, key, now)
    return gate, keys


def saturated() -> Growth:
    return measured(
        lambda: run_saturated(MAX_TRACKED_CLIENTS, SATURATING, 1),
        f"cap full, {MAX_TRACKED_CLIENTS:,} keys, all buckets",
    )


def big_count_extra() -> tuple[Growth, Growth]:
    small = measured(lambda: run_saturated(BIG_COUNT_SAMPLE, BIG_COUNTS, 1), f"{BIG_COUNT_SAMPLE} keys, counts 1")
    big = measured(
        lambda: run_saturated(BIG_COUNT_SAMPLE, BIG_COUNTS, BIG_COUNT_REPEAT),
        f"{BIG_COUNT_SAMPLE} keys, counts {BIG_COUNT_REPEAT}",
    )
    return small, big


def run_sessions(driven: int, message_char: str) -> tuple[Gate, int]:
    """`driven` sessions each with HISTORY_EXCHANGES full exchanges; the cap keeps MAX_SESSIONS."""
    gate = Gate(gate_limits(ONE_PER_WINDOW))
    ids: list[str] = []
    for n in range(driven):
        now = n / 100
        view = gate.admit(None, generate=False, now=now)
        assert isinstance(view, Stop)
        ids.append(view.session.id)
        for turn in range(HISTORY_EXCHANGES):
            tag = f"{n:07d}{turn}"
            gate.remember(
                view.session.id,
                Exchange(
                    message=tag + message_char * (MESSAGE_CHARS - len(tag)),
                    answer=tag + "a" * (ANSWER_CHARS - len(tag)),
                ),
                now,
            )
    end = driven / 100
    newest = gate.admit(ids[-1], generate=False, now=end)
    assert isinstance(newest, Generate | Stop) and len(newest.session.history) == HISTORY_EXCHANGES
    oldest = gate.admit(ids[0], generate=False, now=end)
    assert isinstance(oldest, Stop) and oldest.session.id != ids[0], "the cap kept a session past it"
    return gate, min(driven, MAX_SESSIONS)


def sessions(message_char: str) -> Growth:
    label = "ASCII" if message_char.isascii() else f"U+{ord(message_char):04X}"
    return measured(lambda: run_sessions(2 * MAX_SESSIONS, message_char), f"sessions at cap, {label} messages")


def rss_bytes() -> int:
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


SCENARIOS: Final[dict[str, Callable[[], object]]] = {
    "flood": lambda: run_flood(FLOOD_KEYS),
    "saturated": lambda: run_saturated(MAX_TRACKED_CLIENTS, SATURATING, 1),
    "sessions": lambda: run_sessions(2 * MAX_SESSIONS, "q"),
}


def rss_delta(scenario: str) -> int:
    """Runs one scenario in a fresh process without tracemalloc and returns its RSS growth."""
    out = subprocess.run(
        [sys.executable, "-m", "bench.memory", "--rss", scenario],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(json.loads(out)["rss_delta_bytes"])


def mb(n: float) -> str:
    return f"{n / 1_000_000:.2f} MB"


def report() -> None:
    print(f"MAX_TRACKED_CLIENTS={MAX_TRACKED_CLIENTS:,}  budget={mb(BUDGET_BYTES)}  python={sys.version.split()[0]}")
    rows = [flood(FLOOD_KEYS), saturated()]
    small, big = big_count_extra()
    extra_per_key = (big.growth_bytes - small.growth_bytes) / BIG_COUNT_SAMPLE
    rows += [small, big, sessions("q"), sessions("\U0001f600")]
    for row in rows:
        print(
            f"{row.scenario:42}  held={row.held:>7,}  growth={mb(row.growth_bytes):>9}  "
            f"peak={mb(row.peak_bytes):>9}  per item={row.per_item_bytes:>8.1f} B"
        )
    worst = rows[1].growth_bytes + MAX_TRACKED_CLIENTS * extra_per_key
    print(
        f"counts above 256 add {extra_per_key:.1f} B per key; worst case (cap full, every count unshared) = {mb(worst)}"
    )
    for scenario in SCENARIOS:
        print(f"RSS delta, {scenario:10} (no tracemalloc): {mb(rss_delta(scenario))}")


def main(argv: list[str]) -> None:
    if argv[:1] == ["--rss"]:
        before = rss_bytes()
        kept = SCENARIOS[argv[1]]()
        print(json.dumps({"rss_delta_bytes": rss_bytes() - before}))
        del kept
        return
    report()


if __name__ == "__main__":
    main(sys.argv[1:])

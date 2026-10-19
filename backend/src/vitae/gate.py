"""Every piece of shared mutable request state: rate counters, sessions, slots and the queue."""

import asyncio
import math
import secrets
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from ipaddress import IPv4Address
from typing import Final, Self

from vitae.contract import FinalRejection, Rejection, RetryableRejection, SessionId
from vitae.settings import ClientKey, RateLimit, Settings

HISTORY_EXCHANGES: Final = 3
BUCKETS_PER_WINDOW: Final = 6
_RING: Final = BUCKETS_PER_WINDOW + 1
MAX_TRACKED_CLIENTS: Final = 10_000
"""Hard cap on keys held at once. Sized by the T5 probe (`python -m bench.memory`) to stay under
10 MB with every key live and every bucket non-zero. Past it, the least recently seen key is dropped."""
SWEEP_BUDGET: Final = 2
"""Stale entries dropped per call, so stale keys leave at least twice as fast as new ones arrive."""
GUARD_WORKERS: Final = 2
"""Threads that run screening. Also the global bound on classifier CPU."""
MAX_SCREENING: Final = GUARD_WORKERS * 4
"""Requests screened or waiting for a guard thread at once. Past it, busy before any CPU is spent."""
BUSY_RETRY_S: Final = 5
BUSY_DETAIL: Final = "Every generation slot and queue place is taken."
SCREENING_DETAIL: Final = "Too many requests are being screened right now."
RATE_DETAIL: Final = "This address has sent too many requests."
SESSION_DETAIL: Final = "This session has used all its turns. Start a new conversation."


@dataclass(frozen=True, slots=True)
class GateLimits:
    rate_limits: tuple[RateLimit, ...]
    slots: int
    queue_size: int
    session_max_turns: int
    session_idle_s: int
    max_sessions: int

    @classmethod
    def of(cls, settings: Settings) -> Self:
        return cls(
            rate_limits=settings.rate_limits,
            slots=settings.generation_slots,
            queue_size=settings.queue_size,
            session_max_turns=settings.session_max_turns,
            session_idle_s=settings.session_idle_s,
            max_sessions=settings.max_sessions,
        )


@dataclass(frozen=True, slots=True)
class Exchange:
    message: str
    """The normalized visitor message."""
    answer: str
    """Released sentences only, exactly what the visitor saw."""


@dataclass(frozen=True, slots=True)
class SessionView:
    id: SessionId
    turns_left: int
    history: tuple[Exchange, ...]
    """Oldest first, at most HISTORY_EXCHANGES."""


class Lease:
    """A reserved generation place: a slot now, or a queue place that becomes one."""

    __slots__ = ("_capacity", "_released", "_waiter")

    def __init__(self, capacity: "_Capacity", waiter: "_Waiter") -> None:
        self._capacity = capacity
        self._waiter = waiter
        self._released = False

    async def positions(self) -> AsyncGenerator[int]:
        """Yields the queue position (1 = next) when first queued and whenever it changes, and
        returns when the slot is granted. Returns at once if a slot was free at admit."""
        last = 0
        while not (self._waiter.granted or self._released):
            position, changed = self._capacity.place(self._waiter)
            if position != last:
                yield position
                last = position
            await changed.wait()

    def release(self) -> None:
        """Idempotent. A held slot passes straight to the head waiter (no barging by new arrivals);
        a queue place is removed and everyone behind moves up."""
        if not self._released:
            self._released = True
            self._capacity.release(self._waiter)


@dataclass(frozen=True, slots=True)
class Generate:
    """Admitted to generate. Always holds a lease, so "generate without a place" cannot be built."""

    session: SessionView
    lease: Lease


@dataclass(frozen=True, slots=True)
class Stop:
    """Admitted to answer with a refusal or the availability answer. Never holds capacity."""

    session: SessionView


Admitted = Generate | Stop


class ScreenPermit:
    """One place among the requests being screened right now. Idempotent release()."""

    __slots__ = ("_on_release",)

    def __init__(self, on_release: Callable[[], None]) -> None:
        self._on_release: Callable[[], None] | None = on_release

    def release(self) -> None:
        if self._on_release is not None:
            self._on_release()
            self._on_release = None


class Gate:
    """Every piece of shared mutable request state, with one owner. Every method is sync and is
    called only from the event loop thread, so each call is atomic without a lock. `now` is a
    monotonic timestamp passed in, so tests drive time without patching."""

    _limiter: "_RateLimiter"
    _sessions: "_Sessions"
    _capacity: "_Capacity"
    _screening: int
    """Requests holding a ScreenPermit."""

    def __init__(self, limits: GateLimits) -> None:
        self._limiter = _RateLimiter(limits.rate_limits)
        self._sessions = _Sessions(limits.session_max_turns, limits.session_idle_s, limits.max_sessions)
        self._capacity = _Capacity(limits.slots, limits.queue_size)
        self._screening = 0

    def throttle(self, key: ClientKey, session_id: SessionId | None, now: float) -> ScreenPermit | Rejection:
        """Everything that must hold before any classifier time is spent, in one atomic call:
        1. rate_limited if any per-key window is full; otherwise the request is counted.
        2. session_limit if session_id names a live session with no turns left (read-only, nothing spent).
        3. busy if MAX_SCREENING requests are already being screened; otherwise a ScreenPermit.
        A flood from one key cannot buy classifier time (1), an exhausted session cannot (2), and a
        flood across many keys cannot queue unbounded classifier work (3)."""
        limited = self._limiter.hit(key, now)
        if limited is not None:
            return limited
        if session_id is not None and self._sessions.exhausted(session_id, now):
            return FinalRejection(code="session_limit", detail=SESSION_DETAIL)
        if self._screening >= MAX_SCREENING:
            return RetryableRejection(code="busy", retry_after_s=BUSY_RETRY_S, detail=SCREENING_DETAIL)
        self._screening += 1
        return ScreenPermit(self._screening_done)

    def _screening_done(self) -> None:
        self._screening -= 1

    def admit(self, session_id: SessionId | None, *, generate: bool, now: float) -> Admitted | Rejection:
        """Check, then commit, in one call. An unknown or expired id starts a new session with a
        server-made id (the client never chooses one). session_limit is checked again here, because a
        concurrent request may have spent the last turn while this one was screened. With
        generate=True, busy when every slot and queue place is taken. On any rejection nothing
        changes. On success the turn is counted, and the result is Generate (holding a reserved place)
        or Stop."""
        if session_id is not None and self._sessions.exhausted(session_id, now):
            return FinalRejection(code="session_limit", detail=SESSION_DETAIL)
        if not generate:
            return Stop(self._sessions.spend(session_id, now))
        lease = self._capacity.reserve()
        if lease is None:
            return RetryableRejection(code="busy", retry_after_s=BUSY_RETRY_S, detail=BUSY_DETAIL)
        return Generate(self._sessions.spend(session_id, now), lease)

    def remember(self, session_id: SessionId, exchange: Exchange, now: float) -> None:
        """Appends to history, keeping the newest HISTORY_EXCHANGES. A session evicted while the turn
        ran is ignored. Refused and availability turns are never remembered, so an injection
        attempt never re-enters a later prompt."""
        self._sessions.remember(session_id, exchange, now)


class _Window:
    """One RateLimit's counts for one key: BUCKETS_PER_WINDOW + 1 small ints in a ring, newest at
    `epoch`. The window (now - W, now] always reaches into the bucket K back, so that whole bucket
    is counted too: the limiter never admits more than N in any real window, and holds a client at
    most W/K longer than an exact log would."""

    __slots__ = ("counts", "epoch")

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.counts = [0] * _RING

    def roll(self, epoch: int) -> None:
        """Zeroes the buckets that slid out, O(K)."""
        if epoch - self.epoch >= _RING:
            self.counts = [0] * _RING
        else:
            for bucket in range(self.epoch + 1, epoch + 1):
                self.counts[bucket % _RING] = 0
        self.epoch = epoch


def _bucket(now: float, limit: RateLimit) -> int:
    return int(now * BUCKETS_PER_WINDOW // limit.window_s)


def _retry_after(window: _Window, limit: RateLimit, now: float) -> int:
    """Seconds until enough of the oldest buckets slide out for one more request to fit."""
    total = sum(window.counts)
    oldest = window.epoch - BUCKETS_PER_WINDOW
    for bucket in range(oldest, window.epoch + 1):
        total -= window.counts[bucket % _RING]
        if total < limit.requests:
            free_at = (bucket + _RING) * limit.window_s / BUCKETS_PER_WINDOW
            return max(1, math.ceil(free_at - now))
    raise AssertionError("a full window always drains")


def _pack(key: ClientKey) -> int:
    if isinstance(key, IPv4Address):
        return int(key)
    return int(key.network_address) >> 64 | 1 << 64


class _RateLimiter:
    """Keys are packed to ints: IPv4 -> its 32-bit int, IPv6 /64 -> (prefix >> 64) | 1 << 64. Ints are
    ~4x smaller than the ipaddress objects and hash faster. Insertion order of _clients is recency
    order: a touch pops and reinserts, so the front is always the stalest key."""

    _clients: dict[int, tuple[float, list[_Window]]]

    def __init__(self, limits: tuple[RateLimit, ...]) -> None:
        self._limits = limits
        self._forget_after = max(limit.window_s for limit in limits) * _RING / BUCKETS_PER_WINDOW
        self._clients = {}

    def hit(self, key: ClientKey, now: float) -> Rejection | None:
        self._sweep(now)
        packed = _pack(key)
        entry = self._clients.pop(packed, None)
        if entry is None:
            if len(self._clients) >= MAX_TRACKED_CLIENTS:
                del self._clients[next(iter(self._clients))]
            windows = [_Window(_bucket(now, limit)) for limit in self._limits]
        else:
            windows = entry[1]
        self._clients[packed] = (now, windows)
        retry = 0
        for limit, window in zip(self._limits, windows):
            window.roll(_bucket(now, limit))
            if sum(window.counts) >= limit.requests:
                retry = max(retry, _retry_after(window, limit, now))
        if retry:
            return RetryableRejection(code="rate_limited", retry_after_s=retry, detail=RATE_DETAIL)
        for window in windows:
            window.counts[window.epoch % _RING] += 1
        return None

    def _sweep(self, now: float) -> None:
        """A key unseen for the longest window plus one bucket has every bucket empty, so dropping
        it changes no decision."""
        for _ in range(SWEEP_BUDGET):
            stalest = next(iter(self._clients.items()), None)
            if stalest is None or stalest[1][0] > now - self._forget_after:
                return
            del self._clients[stalest[0]]


@dataclass(slots=True)
class _Session:
    turns_used: int
    last_seen: float
    history: tuple[Exchange, ...]


class _Sessions:
    """Recency-ordered like _RateLimiter: a sweep pops idle sessions from the front, and at
    max_sessions the front is evicted."""

    _sessions: dict[SessionId, _Session]

    def __init__(self, max_turns: int, idle_s: int, max_sessions: int) -> None:
        self._max_turns = max_turns
        self._idle_s = idle_s
        self._max_sessions = max_sessions
        self._sessions = {}

    def exhausted(self, session_id: SessionId, now: float) -> bool:
        session = self._live(session_id, now)
        return session is not None and session.turns_used >= self._max_turns

    def spend(self, session_id: SessionId | None, now: float) -> SessionView:
        """Counts one turn on the live session, or on a new one when the id is unknown or expired."""
        self._sweep(now)
        found = self._sessions.pop(session_id, None) if session_id is not None else None
        if session_id is None or found is None or self._expired(found, now):
            sid = secrets.token_urlsafe(16)
            session = _Session(turns_used=0, last_seen=now, history=())
            if len(self._sessions) >= self._max_sessions:
                del self._sessions[next(iter(self._sessions))]
        else:
            sid, session = session_id, found
        session.turns_used += 1
        session.last_seen = now
        self._sessions[sid] = session
        return SessionView(id=sid, turns_left=self._max_turns - session.turns_used, history=session.history)

    def remember(self, session_id: SessionId, exchange: Exchange, now: float) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.history = (*session.history, exchange)[-HISTORY_EXCHANGES:]
        session.last_seen = now
        self._sessions[session_id] = session

    def _live(self, session_id: SessionId, now: float) -> _Session | None:
        session = self._sessions.get(session_id)
        return None if session is None or self._expired(session, now) else session

    def _expired(self, session: _Session, now: float) -> bool:
        return now - session.last_seen >= self._idle_s

    def _sweep(self, now: float) -> None:
        for _ in range(SWEEP_BUDGET):
            stalest = next(iter(self._sessions.items()), None)
            if stalest is None or not self._expired(stalest[1], now):
                return
            del self._sessions[stalest[0]]


class _Waiter:
    __slots__ = ("granted",)

    def __init__(self, granted: bool) -> None:
        self.granted = granted


class _Capacity:
    _free: int
    """Slots not held."""
    _waiting: deque[_Waiter]
    """FIFO. A waiter's position is its index + 1, derived and never stored."""
    _changed: asyncio.Event
    """Replaced on every change; waiters await the one they hold."""

    def __init__(self, slots: int, queue_size: int) -> None:
        self._free = slots
        self._queue_size = queue_size
        self._waiting = deque()
        self._changed = asyncio.Event()

    def reserve(self) -> Lease | None:
        if self._free > 0:
            self._free -= 1
            return Lease(self, _Waiter(granted=True))
        if len(self._waiting) < self._queue_size:
            waiter = _Waiter(granted=False)
            self._waiting.append(waiter)
            return Lease(self, waiter)
        return None

    def place(self, waiter: _Waiter) -> tuple[int, asyncio.Event]:
        """The waiter's position and the event set on the next change, read together so a change
        after this call always wakes whoever awaits the event."""
        return self._waiting.index(waiter) + 1, self._changed

    def release(self, waiter: _Waiter) -> None:
        if not waiter.granted:
            self._waiting.remove(waiter)
        elif self._waiting:
            self._waiting.popleft().granted = True
        else:
            self._free += 1
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

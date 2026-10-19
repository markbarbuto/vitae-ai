import asyncio
import copy
from fractions import Fraction
from ipaddress import IPv4Address, IPv6Network, ip_address

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, initialize, invariant, rule

from vitae.contract import FinalRejection, Rejection, RetryableRejection, SessionId
from vitae.gate import (
    BUSY_DETAIL,
    BUSY_RETRY_S,
    HISTORY_EXCHANGES,
    MAX_SCREENING,
    MAX_TRACKED_CLIENTS,
    RATE_DETAIL,
    SCREENING_DETAIL,
    SESSION_DETAIL,
    Admitted,
    Exchange,
    Gate,
    GateLimits,
    Generate,
    Lease,
    ScreenPermit,
    Stop,
)
from vitae.settings import ClientKey, RateLimit, client_key, load_settings

ALICE = IPv4Address("203.0.113.9")
BOB = IPv4Address("203.0.113.10")
SESSION_LIMIT = FinalRejection(code="session_limit", detail=SESSION_DETAIL)
BUSY = RetryableRejection(code="busy", retry_after_s=BUSY_RETRY_S, detail=BUSY_DETAIL)


def limits(
    rate: tuple[RateLimit, ...] = (RateLimit(100, 60),),
    slots: int = 1,
    queue: int = 0,
    turns: int = 20,
    idle_s: int = 1800,
    max_sessions: int = 1000,
) -> GateLimits:
    return GateLimits(
        rate_limits=rate,
        slots=slots,
        queue_size=queue,
        session_max_turns=turns,
        session_idle_s=idle_s,
        max_sessions=max_sessions,
    )


def expect_admitted(result: Admitted | Rejection) -> Admitted:
    assert isinstance(result, Generate | Stop), result
    return result


def expect_generate(result: Admitted | Rejection) -> Generate:
    assert isinstance(result, Generate), result
    return result


def screened(gate: Gate, key: ClientKey, now: float, session_id: SessionId | None = None) -> Rejection | None:
    """Throttles one request and releases its permit at once, as a request that screened instantly."""
    result = gate.throttle(key, session_id, now)
    if isinstance(result, ScreenPermit):
        result.release()
        return None
    return result


def rate_limited(retry_after_s: int) -> RetryableRejection:
    return RetryableRejection(code="rate_limited", retry_after_s=retry_after_s, detail=RATE_DETAIL)


async def first_position(lease: Lease) -> int | None:
    positions = lease.positions()
    try:
        return await anext(positions, None)
    finally:
        await positions.aclose()


def test_limits_come_from_settings():
    settings_ = load_settings(
        {
            "VITAE_ALLOWED_ORIGINS": "https://owner.example",
            "VITAE_RATE_LIMITS": "6/minute,200/day",
            "VITAE_GENERATION_SLOTS": "2",
            "VITAE_QUEUE_SIZE": "3",
            "VITAE_SESSION_MAX_TURNS": "7",
            "VITAE_SESSION_IDLE_MINUTES": "5",
            "VITAE_MAX_SESSIONS": "40",
        }
    )
    assert GateLimits.of(settings_) == GateLimits(
        rate_limits=(RateLimit(6, 60), RateLimit(200, 86400)),
        slots=2,
        queue_size=3,
        session_max_turns=7,
        session_idle_s=300,
        max_sessions=40,
    )


def test_rate_limited_names_the_wait_until_the_oldest_bucket_slides_out():
    gate = Gate(limits(rate=(RateLimit(2, 60),)))
    assert screened(gate, ALICE, 0.0) is None
    assert screened(gate, ALICE, 1.0) is None
    # Both hits sit in the 10 s bucket [0, 10), which is counted until 70 s (exact log: 60 s).
    assert screened(gate, ALICE, 2.0) == rate_limited(68)
    assert screened(gate, BOB, 2.0) is None
    assert screened(gate, ALICE, 69.5) == rate_limited(1)
    assert screened(gate, ALICE, 70.0) is None


def test_rate_limited_waits_for_the_slowest_full_window():
    gate = Gate(limits(rate=(RateLimit(1, 1), RateLimit(2, 3600))))
    assert screened(gate, ALICE, 0.0) is None
    assert screened(gate, ALICE, 0.5) == rate_limited(1)
    assert screened(gate, ALICE, 2.0) is None
    assert screened(gate, ALICE, 3.0) == rate_limited(4197)


def test_rejected_requests_spend_nothing():
    gate = Gate(limits(rate=(RateLimit(1, 60),)))
    assert screened(gate, ALICE, 0.0) is None
    for second in range(1, 60):
        assert isinstance(screened(gate, ALICE, float(second)), RetryableRejection)
    assert screened(gate, ALICE, 70.0) is None


def test_ipv6_clients_share_their_64_and_ipv4_neighbours_do_not():
    gate = Gate(limits(rate=(RateLimit(1, 60),)))
    first = client_key(ip_address("2001:db8:1:2::1"))
    same_64 = client_key(ip_address("2001:db8:1:2:ffff::9"))
    other_64 = client_key(ip_address("2001:db8:1:3::1"))
    assert screened(gate, first, 0.0) is None
    assert screened(gate, same_64, 1.0) == rate_limited(69)
    assert screened(gate, other_64, 1.0) is None
    assert screened(gate, IPv4Address("0.0.0.1"), 1.0) is None
    assert screened(gate, IPv4Address("0.0.0.2"), 1.0) is None


def test_screening_is_bounded_and_permits_release_once():
    gate = Gate(limits())
    permits = [gate.throttle(IPv4Address(n), None, 0.0) for n in range(MAX_SCREENING)]
    assert all(isinstance(permit, ScreenPermit) for permit in permits)
    full = RetryableRejection(code="busy", retry_after_s=BUSY_RETRY_S, detail=SCREENING_DETAIL)
    assert gate.throttle(IPv4Address(999), None, 0.0) == full
    first = permits[0]
    assert isinstance(first, ScreenPermit)
    first.release()
    first.release()
    assert isinstance(gate.throttle(IPv4Address(1000), None, 0.0), ScreenPermit)
    assert gate.throttle(IPv4Address(1001), None, 0.0) == full


def test_session_limit_counts_refusals_and_busy_does_not():
    gate = Gate(limits(slots=1, queue=0, turns=2))
    sid = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    expect_generate(gate.admit(sid, generate=True, now=1.0))
    assert gate.admit(sid, generate=True, now=2.0) == SESSION_LIMIT
    other = expect_admitted(gate.admit(None, generate=False, now=2.0))
    assert gate.admit(other.session.id, generate=True, now=3.0) == BUSY
    after = expect_admitted(gate.admit(other.session.id, generate=False, now=4.0))
    assert after.session.turns_left == 0
    assert after.session.id == other.session.id


def test_new_sessions_get_a_server_made_id_and_full_turns():
    gate = Gate(limits(turns=5))
    first = expect_admitted(gate.admit(None, generate=False, now=0.0)).session
    assert len(first.id) == 22
    assert first.turns_left == 4
    assert first.history == ()
    unknown = expect_admitted(gate.admit("A" * 22, generate=False, now=0.0)).session
    assert unknown.id not in ("A" * 22, first.id)
    assert unknown.turns_left == 4


def test_busy_creates_no_session():
    gate = Gate(limits(slots=1, queue=0, max_sessions=2))
    holder = expect_generate(gate.admit(None, generate=True, now=0.0))
    kept = expect_admitted(gate.admit(None, generate=False, now=1.0)).session.id
    for second in range(2, 10):
        assert gate.admit(None, generate=True, now=float(second)) == BUSY
    assert expect_admitted(gate.admit(kept, generate=False, now=10.0)).session.id == kept
    assert expect_admitted(gate.admit(holder.session.id, generate=False, now=10.0)).session.id == holder.session.id


def test_throttle_session_check_is_read_only_and_admit_checks_again():
    gate = Gate(limits(turns=2))
    sid = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    assert screened(gate, ALICE, 1.0, sid) is None
    assert screened(gate, BOB, 1.0, sid) is None
    assert expect_admitted(gate.admit(sid, generate=False, now=2.0)).session.turns_left == 0
    assert gate.admit(sid, generate=False, now=2.0) == SESSION_LIMIT
    assert screened(gate, ALICE, 3.0, sid) == SESSION_LIMIT
    assert screened(gate, ALICE, 4.0, sid) == SESSION_LIMIT
    assert gate.admit(sid, generate=False, now=5.0) == SESSION_LIMIT


def test_throttle_does_not_keep_a_session_alive():
    gate = Gate(limits(turns=1, idle_s=100))
    sid = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    assert screened(gate, ALICE, 90.0, sid) == SESSION_LIMIT
    assert screened(gate, ALICE, 100.0, sid) is None
    fresh = expect_admitted(gate.admit(sid, generate=False, now=100.0)).session
    assert fresh.id != sid
    assert fresh.turns_left == 0


def test_an_idle_session_expires_and_a_used_one_does_not():
    gate = Gate(limits(turns=20, idle_s=100))
    sid = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    kept = expect_admitted(gate.admit(sid, generate=False, now=99.0)).session
    assert (kept.id, kept.turns_left) == (sid, 18)
    kept = expect_admitted(gate.admit(sid, generate=False, now=198.0)).session
    assert (kept.id, kept.turns_left) == (sid, 17)
    expired = expect_admitted(gate.admit(sid, generate=False, now=298.0)).session
    assert expired.id != sid
    assert expired.turns_left == 19


def test_at_max_sessions_the_least_recently_used_is_evicted():
    gate = Gate(limits(max_sessions=2))
    a = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    b = expect_admitted(gate.admit(None, generate=False, now=1.0)).session.id
    assert expect_admitted(gate.admit(a, generate=False, now=2.0)).session.id == a
    c = expect_admitted(gate.admit(None, generate=False, now=3.0)).session.id
    assert expect_admitted(gate.admit(a, generate=False, now=4.0)).session.id == a
    assert expect_admitted(gate.admit(c, generate=False, now=5.0)).session.id == c
    assert expect_admitted(gate.admit(b, generate=False, now=6.0)).session.id != b


def test_history_keeps_the_newest_exchanges_oldest_first():
    gate = Gate(limits())
    sid = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    for n in range(HISTORY_EXCHANGES + 2):
        gate.remember(sid, Exchange(message=f"q{n}", answer=f"a{n}"), now=float(n))
    view = expect_admitted(gate.admit(sid, generate=False, now=10.0)).session
    assert view.history == (Exchange("q2", "a2"), Exchange("q3", "a3"), Exchange("q4", "a4"))


def test_remember_ignores_an_evicted_session():
    gate = Gate(limits(max_sessions=1))
    gone = expect_admitted(gate.admit(None, generate=False, now=0.0)).session.id
    kept = expect_admitted(gate.admit(None, generate=False, now=1.0)).session.id
    gate.remember(gone, Exchange("q", "a"), now=2.0)
    assert expect_admitted(gate.admit(kept, generate=False, now=3.0)).session.history == ()
    revived = expect_admitted(gate.admit(gone, generate=False, now=4.0)).session
    assert revived.id != gone
    assert revived.history == ()


def test_a_released_slot_passes_to_the_head_waiter_without_barging():
    gate = Gate(limits(slots=1, queue=2))
    running = expect_generate(gate.admit(None, generate=True, now=0.0)).lease
    second = expect_generate(gate.admit(None, generate=True, now=0.0)).lease
    third = expect_generate(gate.admit(None, generate=True, now=0.0)).lease
    assert gate.admit(None, generate=True, now=0.0) == BUSY
    assert asyncio.run(first_position(running)) is None
    assert asyncio.run(first_position(second)) == 1
    assert asyncio.run(first_position(third)) == 2
    running.release()
    running.release()
    late = expect_generate(gate.admit(None, generate=True, now=1.0)).lease
    assert asyncio.run(first_position(second)) is None
    assert asyncio.run(first_position(third)) == 1
    assert asyncio.run(first_position(late)) == 2
    third.release()
    assert asyncio.run(first_position(late)) == 1
    assert gate.admit(None, generate=True, now=2.0) != BUSY


def test_a_queued_lease_yields_each_position_then_returns_when_granted():
    async def scenario() -> list[int]:
        gate = Gate(limits(slots=1, queue=3))
        leases = [expect_generate(gate.admit(None, generate=True, now=0.0)).lease for _ in range(4)]
        seen: list[int] = []

        async def watch() -> None:
            async for position in leases[3].positions():
                seen.append(position)

        task = asyncio.create_task(watch())
        await asyncio.sleep(0)
        leases[2].release()
        await asyncio.sleep(0)
        leases[0].release()
        leases[1].release()
        await asyncio.wait_for(task, timeout=1)
        return seen

    assert asyncio.run(scenario()) == [3, 2]


def throttle_admits(gate: Gate, key: ClientKey, now: float) -> bool:
    return screened(gate, key, now) is None


def test_past_the_cap_the_least_recently_seen_key_is_dropped():
    gate = Gate(limits(rate=(RateLimit(1, 3600),)))
    keys = [IPv4Address(n) for n in range(MAX_TRACKED_CLIENTS)]
    for key in keys:
        assert throttle_admits(gate, key, 0.0)
    assert not throttle_admits(gate, keys[0], 1.0)
    newcomer = IPv4Address(MAX_TRACKED_CLIENTS)
    assert throttle_admits(gate, newcomer, 2.0)
    assert throttle_admits(gate, keys[1], 3.0), "the stalest key was dropped and starts fresh"
    assert not throttle_admits(gate, keys[0], 4.0), "a touched key keeps its count"
    assert not throttle_admits(gate, newcomer, 4.0)
    assert throttle_admits(gate, keys[2], 5.0), "the next stalest goes when keys[1] returned"
    assert not throttle_admits(gate, keys[4], 6.0)


WINDOWS = (1, 6, 60, 3600)
KEYS: tuple[ClientKey, ...] = (ALICE, BOB, IPv6Network("2001:db8::/64"), IPv4Address("0.0.0.0"))

times = st.one_of(
    st.integers(0, 64).map(lambda n: Fraction(n, 8)),
    st.integers(0, 8000).map(Fraction),
)


class LimiterAgainstExactLog(RuleBasedStateMachine):
    """The reference is an exact sliding log of the requests the limiter admitted. Times are
    multiples of 1/8 s, so the float clock the gate sees is exact."""

    def __init__(self) -> None:
        super().__init__()
        self.now = Fraction(0)
        self.log: dict[ClientKey, list[Fraction]] = {key: [] for key in KEYS}

    @initialize(
        limits_=st.lists(
            st.tuples(st.integers(1, 4), st.sampled_from(WINDOWS)), min_size=1, max_size=3, unique_by=lambda x: x[1]
        )
    )
    def build(self, limits_: list[tuple[int, int]]) -> None:
        self.limits = tuple(RateLimit(requests, window) for requests, window in limits_)
        self.gate = Gate(limits(rate=self.limits))

    def exact_count(self, key: ClientKey, since: Fraction) -> int:
        return sum(1 for at in self.log[key] if at > since)

    @rule(dt=times)
    def advance(self, dt: Fraction) -> None:
        self.now += dt

    @rule(key=st.sampled_from(KEYS))
    def hit(self, key: ClientKey) -> None:
        before = copy.deepcopy(self.gate)
        result = screened(self.gate, key, float(self.now))
        if result is None:
            for limit in self.limits:
                assert self.exact_count(key, self.now - limit.window_s) < limit.requests, "sound"
            self.log[key].append(self.now)
            return
        assert isinstance(result, RetryableRejection)
        assert result.code == "rate_limited"
        assert any(
            self.exact_count(key, self.now - limit.window_s - Fraction(limit.window_s, 6)) >= limit.requests
            for limit in self.limits
        ), "held at most W/K past the exact log"
        retry_at = float(self.now + result.retry_after_s)
        assert throttle_admits(copy.deepcopy(before), key, retry_at), "admitted once retry_after_s has passed"
        assert result.retry_after_s == 1 or not throttle_admits(before, key, retry_at - 1), "the shortest whole wait"


TestLimiterAgainstExactLog = LimiterAgainstExactLog.TestCase
TestLimiterAgainstExactLog.settings = settings(max_examples=300, stateful_step_count=60, deadline=None)


class CapacityAgainstList(RuleBasedStateMachine):
    """The reference is a list of running leases and a list of waiting ones. Every queued lease has a
    task iterating positions(), so the test also proves each change wakes the right waiters."""

    leases = Bundle("leases")

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.running: list[Lease] = []
        self.waiting: list[Lease] = []
        self.seen: dict[int, list[int]] = {}
        self.watchers: dict[int, asyncio.Task[None]] = {}

    @initialize(slots=st.integers(1, 3), queue=st.integers(0, 4))
    def build(self, slots: int, queue: int) -> None:
        self.slots = slots
        self.queue = queue
        self.gate = Gate(limits(slots=slots, queue=queue))

    def settle(self) -> None:
        for _ in range(4):
            self.loop.run_until_complete(asyncio.sleep(0))

    def watch(self, lease: Lease) -> None:
        seen = self.seen[id(lease)] = []

        async def consume() -> None:
            async for position in lease.positions():
                seen.append(position)

        self.watchers[id(lease)] = self.loop.create_task(consume())

    @rule(target=leases)
    def admit(self) -> Lease | None:
        result = self.gate.admit(None, generate=True, now=0.0)
        if len(self.running) == self.slots and len(self.waiting) == self.queue:
            assert result == BUSY
            return None
        lease = expect_generate(result).lease
        if len(self.running) < self.slots:
            self.running.append(lease)
        else:
            self.waiting.append(lease)
        self.watch(lease)
        self.settle()
        return lease

    @rule(lease=consumes(leases))
    def release(self, lease: Lease | None) -> None:
        if lease is None:
            return
        if lease in self.running:
            self.running.remove(lease)
            if self.waiting:
                self.running.append(self.waiting.pop(0))
        else:
            self.waiting.remove(lease)
        lease.release()
        lease.release()
        self.settle()

    @invariant()
    def matches_the_reference(self) -> None:
        if not hasattr(self, "gate"):
            return
        assert len(self.running) <= self.slots
        assert len(self.waiting) <= self.queue
        assert not self.waiting or len(self.running) == self.slots, "no free slot while anyone waits"
        for lease in self.running:
            assert self.watchers[id(lease)].done()
        for index, lease in enumerate(self.waiting):
            assert not self.watchers[id(lease)].done()
            seen = self.seen[id(lease)]
            assert seen[-1] == index + 1
            assert seen == sorted(set(seen), reverse=True), "each change yielded once, only moving up"

    def teardown(self) -> None:
        for task in self.watchers.values():
            task.cancel()
        self.settle()
        self.loop.close()


TestCapacityAgainstList = CapacityAgainstList.TestCase
TestCapacityAgainstList.settings = settings(max_examples=200, stateful_step_count=40, deadline=None)


def test_limiter_memory_per_tracked_key_fits_the_t5_budget():
    from bench.memory import BUDGET_BYTES, flood

    result = flood(50_000)
    assert result.held == MAX_TRACKED_CLIENTS
    assert result.per_item_bytes <= BUDGET_BYTES / MAX_TRACKED_CLIENTS

from ipaddress import IPv4Address, IPv6Network

from vitae.contract import RetryableRejection
from vitae.gate import RATE_DETAIL, Gate, GateLimits, ScreenPermit
from vitae.settings import RateLimit


def test_an_ipv6_prefix_never_shares_a_counter_with_the_ipv4_address_of_the_same_number():
    gate = Gate(
        GateLimits(
            rate_limits=(RateLimit(1, 60),),
            slots=1,
            queue_size=0,
            session_max_turns=20,
            session_idle_s=1800,
            max_sessions=1000,
        )
    )
    ipv4 = IPv4Address("203.0.113.9")
    ipv6 = IPv6Network("0:0:cb00:7109::/64")
    assert int(ipv6.network_address) >> 64 == int(ipv4)

    first = gate.throttle(ipv4, None, 0.0)
    assert isinstance(first, ScreenPermit)
    first.release()
    assert gate.throttle(ipv4, None, 1.0) == RetryableRejection(
        code="rate_limited", retry_after_s=69, detail=RATE_DETAIL
    )
    assert isinstance(gate.throttle(ipv6, None, 1.0), ScreenPermit)

"""SRE flood driver: many distinct clients against the running app over real HTTP.

`python -m bench.flood` from backend/. Starts `vitae.dev` in its own process with 127.0.0.1 as the
trusted proxy, then plays the proxy: each request carries an X-Real-IP, so every key is a distinct
IPv4 address or IPv6 /64. Each key sends two requests: a message the PII guard refuses, then the same
again, which the per-IP limit (1 per minute) rejects. The second request uses another address in the
same /64, so an IPv6 key only stays one key if the /64 keying holds.

The load comes from several client processes, because one asyncio httpx client saturates before the
server does. Every second the driver samples the server's resident set and the keys done. The
summary is the RSS growth from after a warm-up to the end (the end-to-end check of the T5 gate
probe), the server's CPU per request, and p50/p99 client-side latency per outcome. Client and
server share this machine, so latencies include the clients' own CPU; RSS does not depend on it.
"""

import argparse
import asyncio
import ctypes
import json
import multiprocessing
import os
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address, IPv6Address
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from queue import Empty
from typing import Final

import httpx

from vitae.contract import CHAT_PATH
from vitae.guards.injection import Calibration

BACKEND: Final = Path(__file__).resolve().parents[1]
REPO: Final = BACKEND.parent
ORIGIN: Final = "http://localhost:5173"
REFUSED_MESSAGE: Final = "My email is visitor@example.com, can you write back?"
WARMUP_KEYS: Final = 2000
MILESTONES: Final = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
MiB: Final = 1024 * 1024


def _serve(env: Mapping[str, str], argv: list[str]) -> None:
    """multiprocessing target, so the child's pid is the server's own (a venv launcher would hide it)."""
    from vitae.dev import main

    os.environ.update(env)
    sys.stdout = sys.stderr = open(os.devnull, "w")  # noqa: SIM115 - lives as long as the server
    main(argv)


def rss_bytes(pid: int) -> int:
    """Resident set (working set on Windows) of another process."""
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
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            raise OSError(f"OpenProcess({pid}) failed")
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                raise OSError("GetProcessMemoryInfo failed")
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def cpu_seconds(pid: int) -> float:
    """User plus kernel CPU time another process has used."""
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p, *[ctypes.POINTER(ctypes.c_ulonglong)] * 4]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            raise OSError(f"OpenProcess({pid}) failed")
        try:
            created, exited, kernel, user = (ctypes.c_ulonglong() for _ in range(4))
            if not kernel32.GetProcessTimes(
                handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
            ):
                raise OSError("GetProcessTimes failed")
            return (kernel.value + user.value) / 1e7
        finally:
            kernel32.CloseHandle(handle)
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def addresses(key: int) -> tuple[str, str]:
    """Two client addresses that must share one rate-limit key: even keys are one IPv4 address,
    odd keys two addresses inside one IPv6 /64."""
    if key % 2 == 0:
        address = str(IPv4Address(0x0A000000 + key // 2))
        return address, address
    prefix = (0x20010DB8 << 96) | ((key // 2) << 64)
    return str(IPv6Address(prefix | 1)), str(IPv6Address(prefix | 0xFFFF_0000_0000_0001))


def outcome(response: httpx.Response) -> str:
    if response.status_code == 200:
        return "refused" if "event: refused" in response.text else "answered"
    try:
        return f"{response.status_code} {response.json()['error']['code']}"
    except (ValueError, KeyError, TypeError):
        return f"{response.status_code} other"


Latencies = dict[str, list[float]]


async def flood(base: str, keys: range, concurrency: int, done: "Synchronized[int] | None" = None) -> Latencies:
    """Two requests per key, as the module docstring says. Returns latency in ms per outcome."""
    latencies: defaultdict[str, list[float]] = defaultdict(list[float])
    next_key = iter(keys)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=60) as client:

        async def send(address: str) -> None:
            started = time.perf_counter()
            response = await client.post(
                CHAT_PATH, json={"message": REFUSED_MESSAGE}, headers={"Origin": ORIGIN, "X-Real-IP": address}
            )
            latencies[outcome(response)].append((time.perf_counter() - started) * 1000)

        async def worker() -> None:
            for key in next_key:
                first, second = addresses(key)
                await send(first)
                await send(second)
                if done is not None:
                    with done.get_lock():
                        done.value += 1

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return dict(latencies)


def _client(base: str, keys: range, concurrency: int, done: "Synchronized[int]", results: "Queue[Latencies]") -> None:
    """multiprocessing target: one client process, so several share the load the way separate visitors would."""
    results.put(asyncio.run(flood(base, keys, concurrency, done)))


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def wait_until_up(base: str, server: BaseProcess) -> None:
    deadline = time.monotonic() + 120
    while True:
        try:
            httpx.get(f"{base}/openapi.json", timeout=1)
            return
        except httpx.TransportError:
            if not server.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("vitae.dev did not start") from None
            time.sleep(0.2)


def drive(base: str, server: BaseProcess, keys: int, clients: int, concurrency: int) -> dict[str, object]:
    pid = server.pid
    assert pid is not None
    wait_until_up(base, server)
    asyncio.run(flood(base, range(WARMUP_KEYS), concurrency))
    baseline = rss_bytes(pid)
    context = multiprocessing.get_context("spawn")
    done = context.Value("q", 0)
    results: Queue[Latencies] = context.Queue()
    first = WARMUP_KEYS
    slices = [range(first + n, first + keys, clients) for n in range(clients)]
    workers = [context.Process(target=_client, args=(base, part, concurrency, done, results)) for part in slices]
    cpu_before = cpu_seconds(pid)
    started = time.perf_counter()
    for worker in workers:
        worker.start()
    rows: list[dict[str, float]] = []
    gathered: list[Latencies] = []
    while len(gathered) < clients:
        try:
            gathered.append(results.get(timeout=1.0))
        except Empty:
            pass
        row = {"t_s": round(time.perf_counter() - started, 1), "keys": done.value, "rss_mib": rss_bytes(pid) / MiB}
        rows.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    elapsed_s = time.perf_counter() - started
    server_cpu_s = cpu_seconds(pid) - cpu_before
    for worker in workers:
        worker.join()
    end = rss_bytes(pid)
    latencies: defaultdict[str, list[float]] = defaultdict(list[float])
    for part in gathered:
        for kind, values in part.items():
            latencies[kind] += values
    requests = sum(len(values) for values in latencies.values())
    return {
        "keys": keys,
        "requests": requests,
        "clients": clients,
        "concurrency_per_client": concurrency,
        "elapsed_s": round(elapsed_s, 1),
        "requests_per_s": round(requests / elapsed_s),
        "server_cpu_s": round(server_cpu_s, 1),
        "server_cpu_ms_per_request": round(server_cpu_s * 1000 / requests, 3),
        "server_cores_busy": round(server_cpu_s / elapsed_s, 2),
        "rss_baseline_mib": round(baseline / MiB, 2),
        "rss_end_mib": round(end / MiB, 2),
        "rss_peak_mib": round(max(row["rss_mib"] for row in rows), 2),
        "rss_growth_mib": round((end - baseline) / MiB, 2),
        "rss_mib_at_keys": {
            str(milestone): round(next(row["rss_mib"] for row in rows if row["keys"] >= milestone), 2)
            for milestone in MILESTONES
            if milestone <= max(row["keys"] for row in rows)
        },
        "latency_ms": {
            kind: {
                "count": len(values),
                "p50": round(percentile(values, 0.5), 2),
                "p99": round(percentile(values, 0.99), 2),
            }
            for kind, values in sorted(latencies.items())
        },
        "per_second": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.flood", description="Flood the running app with many distinct clients."
    )
    parser.add_argument("--keys", type=int, default=100_000, help="distinct client keys after the warm-up")
    parser.add_argument("--clients", type=int, default=4, help="client processes")
    parser.add_argument("--concurrency", type=int, default=2, help="requests in flight per client process")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, help="write the full result, per-second rows included, as JSON")
    args = parser.parse_args(argv)

    models = REPO / ".cache" / "models"
    if (problem := Calibration.load().model_problem(models)) is not None:
        print(problem, file=sys.stderr)
        return 1
    env = {
        "VITAE_ALLOWED_ORIGINS": ORIGIN,
        "VITAE_RATE_LIMITS": "1/minute",
        "VITAE_TRUSTED_PROXY_CIDRS": "127.0.0.1/32",
        "VITAE_CLIENT_IP_HEADER": "X-Real-IP",
        "VITAE_PROFILE": str(BACKEND / "profiles" / "example.toml"),
        "VITAE_INJECTION_MODEL_DIR": str(models),
    }
    with tempfile.TemporaryDirectory() as work:
        script = Path(work) / "script.json"
        script.write_text('["I built Shift Notes."]', encoding="utf-8")
        argv_dev = [str(script), "--port", str(args.port)]
        server = multiprocessing.get_context("spawn").Process(target=_serve, args=(env, argv_dev), daemon=True)
        server.start()
        try:
            result = drive(f"http://127.0.0.1:{args.port}", server, args.keys, args.clients, args.concurrency)
        finally:
            server.terminate()
            server.join(10)
    summary = {name: value for name, value in result.items() if name != "per_second"}
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

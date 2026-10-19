"""T7, differential: nginx's realip module (official image, pinned digest) is the judge of the client
address. Generated forwarding-header sets go from client containers on known addresses through nginx
to the app, and the key the app derives is compared with nginx's $remote_addr.

An untrusted client's headers must never move its key. A client in nginx's trusted load-balancer
range must get exactly the key of the address nginx resolved."""

import json
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from ipaddress import IPv6Address, ip_address
from pathlib import Path
from typing import Self

import httpx
import pytest

from tests.http.harness import ORIGIN, build_app, serving, settings_env
from tests.nginx.headers import (
    TRUSTED_V4,
    TRUSTED_V6,
    UNTRUSTED_V4,
    UNTRUSTED_V6,
    Headers,
    header_sets,
)
from vitae.settings import client_key

HERE = Path(__file__).parent
NGINX_IMAGE = "nginx:1.27.3-alpine@sha256:814a8e88df978ade80e584cc5b333144b9372a8e3c98872d07137dbf3b44d0e4"
LABEL = "vitae-t7"
SUBNET_V4 = "172.31.250.0/24"
SUBNET_V6 = "fd00:7a17::/64"
CLIENTS = {"untrusted": (UNTRUSTED_V4, UNTRUSTED_V6), "trusted": (TRUSTED_V4, TRUSTED_V6)}
CHAT_SPOOFS: list[Headers] = [
    [("X-Forwarded-For", "198.51.100.1")],
    [("X-Real-IP", "198.51.100.2")],
    [("Forwarded", "for=198.51.100.3"), ("X-Forwarded-For", f"198.51.100.3, {TRUSTED_V4}")],
    [("X-Forwarded-For", f"2001:db8:9::1, {TRUSTED_V6}"), ("X-Real-IP", "2001:db8:9::1")],
]


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, timeout=300)


def docker_works() -> bool:
    return shutil.which("docker") is not None and docker("info", check=False).returncode == 0


pytestmark = pytest.mark.skipif(not docker_works(), reason="needs a running Docker engine")


@dataclass(frozen=True)
class Request:
    headers: Headers
    body: bytes | None = None
    """None for GET /_test/key; a JSON body for POST /v1/chat."""

    def header_file(self) -> bytes:
        """curl's -H @file format, one header per line. "Name;" is how curl sends an empty value."""
        headers = self.headers if self.body is None else [*CHAT_HEADERS, *self.headers]
        return "\n".join(f"{name}: {value}" if value.strip() else f"{name};" for name, value in headers).encode(
            "latin-1"
        )


CHAT_HEADERS: Headers = [("Origin", ORIGIN), ("Content-Type", "application/json")]


@dataclass(frozen=True)
class Answer:
    status: int
    remote_addr: str | None
    body: bytes


NO_ANSWER = 0
"""The status recorded when curl got no answer at all, so the status assertions name it."""


def parse_response(raw: bytes) -> Answer:
    if not raw:
        return Answer(NO_ANSWER, None, b"")
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line, *header_lines = head.decode("latin-1").split("\r\n")
    headers = {name.lower(): value.strip() for name, _, value in (line.partition(":") for line in header_lines)}
    return Answer(int(status_line.split()[1]), headers.get("x-nginx-remote-addr"), body)


class Stack:
    """A docker network, nginx in front of an app on the host, and one-shot client containers."""

    def __init__(self, work: Path) -> None:
        self.work = work
        self.tag = uuid.uuid4().hex[:8]
        self.network = f"vitae-t7-{self.tag}"
        self.nginx = f"vitae-t7-nginx-{self.tag}"

    def __enter__(self) -> Self:
        self.remove_leftovers()
        docker(
            "network", "create", "--label", LABEL, "--ipv6", "--subnet", SUBNET_V4, "--gateway", "172.31.250.1",
            "--subnet", SUBNET_V6, self.network,
        )  # fmt: skip
        return self

    def __exit__(self, *_: object) -> None:
        self.save_log()
        self.remove_leftovers()

    @staticmethod
    def remove_leftovers() -> None:
        """Containers and networks of this run, or of an earlier run that was killed, by label."""
        containers = docker("ps", "-aq", "--filter", f"label={LABEL}").stdout.split()
        if containers:
            docker("rm", "-f", *containers, check=False)
        docker("network", "prune", "-f", "--filter", f"label={LABEL}", check=False)

    def save_log(self) -> None:
        logs = docker("logs", self.nginx, check=False)
        with (self.work / "nginx.log").open("a", encoding="utf-8") as log:
            log.write(logs.stdout + logs.stderr)

    def start_nginx(self, app_port: int) -> None:
        self.save_log()
        docker("rm", "-f", self.nginx, check=False)
        config = self.work / f"nginx-{app_port}.conf"
        config.write_text((HERE / "nginx.conf").read_text().replace("__APP__", f"host.docker.internal:{app_port}"))
        docker(
            "run", "-d", "--label", LABEL, "--name", self.nginx, "--network", self.network,
            "--ip", "172.31.250.2", "--ip6", "fd00:7a17::2",
            "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{config}:/etc/nginx/nginx.conf:ro", NGINX_IMAGE,
        )  # fmt: skip
        docker("exec", self.nginx, "sh", "-c", "for i in $(seq 50); do nc -z 127.0.0.1 80 && exit 0; sleep 0.1; done")

    def peer_seen_by_app(self) -> str:
        out = docker("exec", self.nginx, "curl", "-s", "http://127.0.0.1/_test/key").stdout
        return json.loads(out)["peer"]

    def send(self, client: str, requests: dict[str, Request]) -> dict[str, Answer]:
        cases, out = self.work / client / "cases", self.work / client / "out"
        cases.mkdir(parents=True)
        out.mkdir(parents=True)
        for name, request in requests.items():
            (cases / f"{name}.hdr").write_bytes(request.header_file())
            if request.body is not None:
                (cases / f"{name}.body").write_bytes(request.body)
        v4, v6 = CLIENTS[client]
        script = (HERE / "client.sh").read_text().replace("\r", "")
        docker(
            "run", "--rm", "--label", LABEL, "--network", self.network, "--ip", v4, "--ip6", v6,
            "-v", f"{cases}:/cases:ro", "-v", f"{out}:/out", NGINX_IMAGE, "sh", "-c", script,
        )  # fmt: skip
        return {name: parse_response((out / f"{name}.resp").read_bytes()) for name in requests}


@dataclass(frozen=True)
class Run:
    sets: list[Headers]
    answers: dict[str, dict[str, Answer]]
    """client -> case name -> answer"""


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
    sets = header_sets()
    with Stack(tmp_path_factory.mktemp("t7")) as stack:
        with serving(build_app(settings_env()), host) as probe:
            stack.start_nginx(httpx.URL(probe).port or 0)
            peer = stack.peer_seen_by_app()
        prefix = 128 if isinstance(ip_address(peer), IPv6Address) else 32
        env = settings_env(
            VITAE_RATE_LIMITS="3/minute",
            VITAE_TRUSTED_PROXY_CIDRS=f"{peer}/{prefix}",
            VITAE_CLIENT_IP_HEADER="X-Real-IP",
        )
        with serving(build_app(env), host) as base:
            stack.start_nginx(httpx.URL(base).port or 0)
            answers: dict[str, dict[str, Answer]] = {}
            for client in CLIENTS:
                requests = {
                    f"key-{n:04d}.{family}": Request(headers)
                    for n, headers in enumerate(sets)
                    for family in ("v4", "v6")
                }
                if client == "untrusted":
                    body = json.dumps({"message": "What projects have you built?"}).encode()
                    requests |= {
                        f"zz-chat-{n}.{family}": Request(headers, body)
                        for n, headers in enumerate(CHAT_SPOOFS)
                        for family in ("v4", "v6")
                    }
                answers[client] = stack.send(client, requests)
        yield Run(sets, answers)


def app_key(answer: Answer) -> str:
    return json.loads(answer.body)["key"]


def own_address(client: str, name: str) -> str:
    v4, v6 = CLIENTS[client]
    return v6 if name.endswith(".v6") else v4


def canonical_key(address: str) -> str:
    parsed = ip_address(address)
    if isinstance(parsed, IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return str(client_key(parsed))


def keyed(run: Run, client: str) -> dict[str, Answer]:
    return {name: answer for name, answer in run.answers[client].items() if name.startswith("key-")}


def test_the_generator_makes_at_least_500_header_sets(run: Run):
    assert len(run.sets) >= 500
    assert len(keyed(run, "untrusted")) == len(keyed(run, "trusted")) == 2 * len(run.sets)


def test_no_header_from_an_untrusted_client_changes_its_key(run: Run):
    answers = keyed(run, "untrusted")
    reached = {name: answer for name, answer in answers.items() if answer.status == 200}
    changed = [
        (name, answer.remote_addr, app_key(answer))
        for name, answer in reached.items()
        if answer.remote_addr != own_address("untrusted", name)
        or app_key(answer) != canonical_key(own_address("untrusted", name))
    ]
    print(
        f"untrusted: {len(answers)} sent, {len(reached)} reached the app, statuses {Counter(a.status for a in answers.values())}"
    )
    assert changed == []
    assert {answer.status for answer in answers.values()} <= {200, 400}


def test_a_trusted_hop_gets_exactly_the_key_of_the_address_nginx_resolved(run: Run):
    answers = keyed(run, "trusted")
    reached = {name: answer for name, answer in answers.items() if answer.status == 200}
    disagree = [
        (name, answer.remote_addr, app_key(answer))
        for name, answer in reached.items()
        if answer.remote_addr is None or app_key(answer) != canonical_key(answer.remote_addr)
    ]
    moved = sum(answer.remote_addr != own_address("trusted", name) for name, answer in reached.items())
    print(f"trusted: {len(answers)} sent, {len(reached)} reached the app, {moved} resolved past the trusted hop")
    assert disagree == []
    assert moved > 100
    assert {answer.status for answer in answers.values()} <= {200, 400}


def test_spoofed_chats_through_nginx_hit_the_untrusted_clients_own_limit(run: Run):
    chats = {name: answer.status for name, answer in run.answers["untrusted"].items() if name.startswith("zz-chat-")}
    for family in ("v4", "v6"):
        assert [chats[f"zz-chat-{n}.{family}"] for n in range(4)] == [200, 200, 200, 429]

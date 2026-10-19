"""Download every file in bench/manifest.json into the repository's .cache/ and verify its sha256.

`python -m bench.fetch` from backend/. Idempotent: a file already present with the right hash is
not downloaded again, and a file with the wrong hash is downloaded again. Exits non-zero if any file
still fails its hash after download.
"""

import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
CACHE: Final = REPO_ROOT / ".cache"
MANIFEST: Final = Path(__file__).with_name("manifest.json")

SourceKind = Literal["hf-dataset", "hf-model", "github"]
_KINDS: Final[tuple[SourceKind, ...]] = ("hf-dataset", "hf-model", "github")
_CHUNK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class Entry:
    kind: SourceKind
    repo: str
    revision: str
    path: str
    sha256: str
    bytes: int
    licence: str

    @property
    def url(self) -> str:
        match self.kind:
            case "github":
                return f"https://raw.githubusercontent.com/{self.repo}/{self.revision}/{self.path}"
            case "hf-dataset":
                return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.path}"
            case "hf-model":
                return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.path}"

    def local(self, cache: Path = CACHE) -> Path:
        """Corpora under .cache/corpora, models under .cache/models, each at <repo>/<path>."""
        area = "models" if self.kind == "hf-model" else "corpora"
        return cache / area / self.repo / self.path


def load_manifest(path: Path = MANIFEST) -> tuple[Entry, ...]:
    entries: list[Entry] = []
    for raw in json.loads(path.read_bytes()):
        kind, _, repo = str(raw["source"]).partition(":")
        if kind not in _KINDS:
            raise ValueError(f"{path}: unknown source kind in {raw['source']!r}")
        entries.append(
            Entry(
                kind=kind,
                repo=repo,
                revision=str(raw["revision"]),
                path=str(raw["path"]),
                sha256=str(raw["sha256"]),
                bytes=int(raw["bytes"]),
                licence=str(raw["licence"]),
            )
        )
    return tuple(entries)


def source_sha256(path: Path) -> str:
    """sha256 of a text file with CRLF read as LF, so a Windows working copy and a fresh LF checkout
    (.gitattributes eol=lf) pin the same digest."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(entry: Entry, cache: Path = CACHE) -> bool:
    local = entry.local(cache)
    return local.is_file() and local.stat().st_size == entry.bytes and sha256_of(local) == entry.sha256


def _download(entry: Entry, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(entry.url) as response, partial.open("wb") as file:
        while chunk := response.read(_CHUNK):
            file.write(chunk)
    partial.replace(dest)


def fetch(entries: tuple[Entry, ...], cache: Path = CACHE) -> tuple[int, list[str]]:
    """Returns how many files were downloaded, and one problem line per file that failed its hash."""
    downloaded = 0
    problems: list[str] = []
    for entry in entries:
        if is_verified(entry, cache):
            continue
        print(f"GET {entry.url}", file=sys.stderr)
        _download(entry, entry.local(cache))
        downloaded += 1
        if not is_verified(entry, cache):
            got = sha256_of(entry.local(cache))
            problems.append(f"{entry.repo}/{entry.path}: expected sha256 {entry.sha256}, got {got}")
    return downloaded, problems


def main() -> int:
    entries = load_manifest()
    downloaded, problems = fetch(entries)
    total_mb = sum(entry.bytes for entry in entries) / 1e6
    print(f"{len(entries)} files ({total_mb:.1f} MB) in {CACHE}: {downloaded} downloaded, {len(problems)} failed")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

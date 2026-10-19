from pathlib import Path

from bench.fetch import Entry, load_manifest


def entry(kind: str, repo: str, path: str) -> Entry:
    return next(e for e in load_manifest() if e.kind == kind and e.repo == repo and e.path == path)


def test_each_source_kind_resolves_to_its_pinned_url_and_cache_area():
    dataset = entry("hf-dataset", "Lakera/gandalf_ignore_instructions", "README.md")
    model = entry("hf-model", "protectai/deberta-v3-base-prompt-injection-v2", "onnx/model.onnx")
    garak = entry("github", "NVIDIA/garak", "garak/data/dan/DUDE.json")
    assert [dataset.url, model.url, garak.url] == [
        "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions/resolve/"
        "04737b65e90a6794ec227012e4a255a7def6344b/README.md",
        "https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/resolve/"
        "90c9989b1a342275dd0d1a95aad283c04e075671/onnx/model.onnx",
        "https://raw.githubusercontent.com/NVIDIA/garak/cc1215c5c301a69f115b0b2b7ef8e4172ae2e7fc/"
        "garak/data/dan/DUDE.json",
    ]
    cache = Path("cache")
    assert [dataset.local(cache), model.local(cache), garak.local(cache)] == [
        cache / "corpora/Lakera/gandalf_ignore_instructions/README.md",
        cache / "models/protectai/deberta-v3-base-prompt-injection-v2/onnx/model.onnx",
        cache / "corpora/NVIDIA/garak/garak/data/dan/DUDE.json",
    ]


def test_every_manifest_entry_pins_a_full_revision_and_hash():
    entries = load_manifest()
    assert len(entries) == 81
    assert all(len(e.revision) == 40 and len(e.sha256) == 64 and e.bytes > 0 and e.licence for e in entries)


def test_source_sha256_reads_crlf_as_lf(tmp_path: Path) -> None:
    from bench.fetch import source_sha256

    (tmp_path / "lf.py").write_bytes(b"a = 1\nb = 2\n")
    (tmp_path / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")

    assert source_sha256(tmp_path / "crlf.py") == source_sha256(tmp_path / "lf.py")
    assert source_sha256(tmp_path / "lf.py") == "fe9c2dadb34bee2ae03fc8fe25e26c64ce2f572a53a127db5d6e73899b64fc11"

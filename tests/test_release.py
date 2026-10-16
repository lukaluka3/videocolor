"""Check the publication boundary without using private source recordings."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_release as release  # noqa: E402
import package_release as packaging  # noqa: E402
import build_notebook as builder  # noqa: E402


def payload(extra: dict[str, str] | None = None) -> bytes:
    stream = io.BytesIO()
    files = {"README.md": "Public instructions", "LICENSE": "Test license", "pyproject.toml": "[project]"}
    files.update(extra or {})
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def notebook(archive: bytes) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"videocolor_payload_sha256": hashlib.sha256(archive).hexdigest()},
        "cells": [{
            "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [f"EMBEDDED_SOURCE_B64 = {base64.b64encode(archive).decode()!r}\n"],
        }],
    }


def test_private_work_files_are_outside_public_selection(tmp_path):
    entries = {
        "README.md": "Instructions", "src/videocolor/cli.py": "pass",
        "src/videocolor/__pycache__/cached.pyc": "cache", "seg_001.mp4": "private",
        "src/recording.mp4": "private", "outputs/run/result.mp4": "private",
        "models/model.pth": "weights", ".env": "secret", "docs/personal.txt": "private",
        "scripts/unreviewed-helper.py": "private", "tests/test_release.py": "pass",
    }
    for name, text in entries.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    names = {p.relative_to(tmp_path).as_posix() for p in release.public_files(tmp_path)}
    assert names == {"README.md", "src/videocolor/cli.py", "tests/test_release.py"}


@pytest.mark.parametrize("private", [
    "person" + "@" + "mail.example",
    "https://colab.research.google.com/" + "drive/" + "private-id",
    "C:" + chr(92) + "Users" + chr(92) + "private-user" + chr(92) + "video.mp4",
    "hf_" + "a" * 30,
])
def test_private_data_is_rejected_without_echoing_it(private):
    with pytest.raises(release.ReleaseError) as caught:
        release.audit_text(private, "README.md")
    assert private not in str(caught.value)


@pytest.mark.parametrize("name", [
    "../outside.py", "/src/injected.py", "src/../injected.py",
    "src//injected.py", "src/video.mp4", "src/weights.pth",
])
def test_notebook_payload_rejects_unsafe_entries(name):
    with pytest.raises(release.ReleaseError):
        release.audit_payload(payload({name: "content"}))


def test_notebook_scans_hidden_payload_for_private_data(tmp_path):
    archive = payload({"src/secret.py": "account = '" + "person" + "@" + "mail.example'"})
    with pytest.raises(release.ReleaseError, match="email address"):
        release.audit_notebook(json.dumps(notebook(archive)).encode(), tmp_path, compare_payload=False)


def test_notebook_rejects_saved_outputs_and_execution_count(tmp_path):
    clean = notebook(payload())
    release.audit_notebook(json.dumps(clean).encode(), tmp_path, compare_payload=False)
    clean["cells"][0]["outputs"] = [{"output_type": "stream", "text": "private"}]
    with pytest.raises(release.ReleaseError, match="outputs"):
        release.audit_notebook(json.dumps(clean).encode(), tmp_path, compare_payload=False)
    clean["cells"][0]["outputs"] = []
    clean["cells"][0]["execution_count"] = 1
    with pytest.raises(release.ReleaseError, match="counts"):
        release.audit_notebook(json.dumps(clean).encode(), tmp_path, compare_payload=False)


def test_notebook_rejects_wrong_hash(tmp_path):
    clean = notebook(payload())
    clean["metadata"]["videocolor_payload_sha256"] = "0" * 64
    with pytest.raises(release.ReleaseError, match="hash"):
        release.audit_notebook(json.dumps(clean).encode(), tmp_path, compare_payload=False)


def test_notebook_rejects_hidden_metadata(tmp_path):
    clean = notebook(payload())
    clean["cells"][0]["metadata"]["executionInfo"] = {"user": "private"}
    with pytest.raises(release.ReleaseError, match="metadata"):
        release.audit_notebook(json.dumps(clean).encode(), tmp_path, compare_payload=False)
    clean["cells"][0]["metadata"] = {}
    clean["account"] = "private"
    with pytest.raises(release.ReleaseError, match="top-level"):
        release.audit_notebook(json.dumps(clean).encode(), tmp_path, compare_payload=False)


def write_builder(root, archive):
    target = root / "scripts" / "build_notebook.py"
    target.parent.mkdir(exist_ok=True)
    target.write_text(f"def payload_bytes():\n    return {archive!r}\n", encoding="utf-8")


def test_notebook_comparison_ignores_archive_encoding_and_git_newlines(tmp_path):
    original = payload({"src/example.py": "answer = 42\r\n"})
    # A Linux checkout/rebuild can use another ZIP order, platform, compression,
    # timestamp and newline convention without changing the application source.
    other = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(other, "w") as target:
        for name in reversed(source.namelist()):
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(entry, source.read(name).replace(b"\r\n", b"\n"))
    rebuilt = other.getvalue()
    assert rebuilt != original
    write_builder(tmp_path, rebuilt)
    release.audit_notebook(json.dumps(notebook(original)).encode(), tmp_path)


def test_notebook_comparison_still_rejects_changed_source(tmp_path):
    original = payload({"src/example.py": "answer = 42\n"})
    write_builder(tmp_path, payload({"src/example.py": "answer = 43\n"}))
    with pytest.raises(release.ReleaseError, match="stale source"):
        release.audit_notebook(json.dumps(notebook(original)).encode(), tmp_path)


def test_notebook_builder_omits_build_metadata_and_normalizes_newlines(tmp_path, monkeypatch):
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "DISCLAIMER.md"):
        (tmp_path / name).write_bytes(b"public\r\n")
    module = tmp_path / "src" / "videocolor" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"value = 1\r\n")
    metadata = tmp_path / "src" / "videocolor_local.egg-info" / "requires.txt"
    metadata.parent.mkdir()
    metadata.write_text("build-only metadata", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    first = builder.payload_bytes()
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert not any(".egg-info" in name for name in archive.namelist())
        assert archive.read("src/videocolor/__init__.py") == b"value = 1\n"
        assert archive.namelist() == sorted(archive.namelist())
        assert all(entry.create_system == 3 for entry in archive.infolist())
    for path in tmp_path.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    assert builder.payload_bytes() == first


def test_existing_release_is_never_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "release.zip"
    target.write_bytes(b"existing archive")
    monkeypatch.setattr(packaging, "audited_contents", lambda root: {"README.md": b"new"})
    with pytest.raises(release.ReleaseError, match="already exists"):
        packaging.package_release(target, tmp_path)
    assert target.read_bytes() == b"existing archive"


def test_package_contains_exact_audited_bytes(tmp_path, monkeypatch):
    approved = {"README.md": b"approved", "src/videocolor/cli.py": b"pass\n"}
    monkeypatch.setattr(packaging, "audited_contents", lambda root: approved)
    target = tmp_path / "release.zip"
    packaging.package_release(target, tmp_path)
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == set(approved)
        assert all(archive.read(name) == content for name, content in approved.items())

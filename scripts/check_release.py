"""Audit the explicit public-source selection without reading private media."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "VideoGaferadeba_Colab.ipynb"
ROOT_FILES = frozenset({
    "README.md", "LICENSE", "DISCLAIMER.md", "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "pyproject.toml",
    ".gitignore", ".gitattributes", "MANIFEST.in", NOTEBOOK,
})
SCRIPT_FILES = frozenset({
    "build_notebook.py", "check_release.py", "package_release.py",
    "video-info.ps1", "cut-clip.ps1",
})
DOC_FILES = frozenset({"usage.md", "limitations.md", "publishing.md", "validation.md"})
IGNORED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv"})
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024

# Patterns identify categories without storing a particular user's private data.
PRIVATE_PATTERNS = (
    ("personal home path", re.compile(r"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/\s\"'<>]+|/(?:Users|home)/[A-Za-z0-9_.-]+)")),
    ("personal notebook or Drive link", re.compile(r"(?i)(?:colab\.research\.google\.com/drive/|drive\.google\.com/(?:file/d/|drive/(?:u/\d+/)?folders/))")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("access token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b")),
)


class ReleaseError(RuntimeError):
    """The selected source cannot safely be packaged."""


def allowed_relative(relative: PurePosixPath) -> bool:
    """Allow source types and known entrypoints, never media or checkpoints."""
    parts = relative.parts
    if not parts or relative.is_absolute() or ".." in parts or any(p in IGNORED_PARTS for p in parts):
        return False
    if len(parts) == 1:
        return parts[0] in ROOT_FILES
    if parts[0] == "src":
        return relative.suffix == ".py" or relative.name == "NOTICE.md" or (
            relative.name.startswith("LICENSE-") and relative.suffix == ".txt"
        )
    if parts[0] == "tests":
        return relative.suffix == ".py"
    if parts[0] == "scripts":
        return len(parts) == 2 and parts[1] in SCRIPT_FILES
    if parts[0] == "docs":
        return len(parts) == 2 and parts[1] in DOC_FILES
    if parts[0] == ".github":
        return relative.suffix in {".md", ".yml", ".yaml"}
    return False


def public_files(root: Path = ROOT) -> list[Path]:
    """Enumerate only allowed source files, refusing links inside the selection."""
    root = root.resolve()
    selected: list[Path] = []
    for name in sorted(ROOT_FILES):
        candidate = root / name
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ReleaseError(f"Source entry must not be a symbolic link: {name}")
        if candidate.is_file():
            selected.append(candidate)
    for name in ("src", "tests", "scripts", "docs", ".github"):
        folder = root / name
        if folder.is_symlink() or not folder.resolve().is_relative_to(root):
            raise ReleaseError(f"Source directory must not be a symbolic link: {name}")
        if not folder.is_dir():
            continue
        for candidate in folder.rglob("*"):
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
                raise ReleaseError(f"Source tree contains a symbolic link: {relative}")
            if candidate.is_file() and allowed_relative(relative):
                selected.append(candidate)
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def audit_text(text: str, label: str) -> None:
    for description, pattern in PRIVATE_PATTERNS:
        if pattern.search(text):
            # Do not echo the matched secret into logs.
            raise ReleaseError(f"Review {label}: detected {description}.")


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
        raise ReleaseError("Notebook contains an invalid cell source.")
    return "".join(value)


def _embedded_string(cells: list[dict]) -> str:
    found: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse(_source(cell))
        except SyntaxError as exc:
            raise ReleaseError("Notebook cells must contain valid Python, without shell magics.") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "EMBEDDED_SOURCE_B64" for target in node.targets
            ):
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    raise ReleaseError("Notebook source payload must be a literal string.")
                found.append(node.value.value)
    if len(found) != 1:
        raise ReleaseError("Notebook must contain exactly one embedded source payload.")
    return found[0]


def audit_payload(payload: bytes) -> dict[str, bytes]:
    """Validate source entries and return text normalized for Git checkouts.

    Archive order, compression, creator OS, and CRLF versus LF are not source
    changes. The notebook's separate SHA-256 check still covers its exact ZIP.
    """
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ReleaseError("Notebook payload exceeds the source archive size limit.")
    paths: dict[str, bytes] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for entry in archive.infolist():
                name = entry.filename
                relative = PurePosixPath(name)
                if ("\\" in name or ":" in name or name != relative.as_posix() or name in paths
                        or entry.is_dir() or not allowed_relative(relative)):
                    raise ReleaseError("Notebook payload contains an unexpected, duplicate, or unsafe path.")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ReleaseError("Notebook payload contains a symbolic link.")
                if name == NOTEBOOK or relative.parts[0] in {"tests", "scripts", "docs", ".github"}:
                    raise ReleaseError("Notebook payload must contain application source and its notices only.")
                total += entry.file_size
                if entry.file_size > MAX_TEXT_BYTES or total > MAX_PAYLOAD_BYTES:
                    raise ReleaseError("Notebook payload has excessive expanded size.")
                content = archive.read(entry).decode("utf-8")
                audit_text(content, f"notebook payload/{name}")
                paths[name] = content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except (zipfile.BadZipFile, UnicodeDecodeError, RuntimeError) as exc:
        if isinstance(exc, ReleaseError):
            raise
        raise ReleaseError("Notebook payload is not a readable source ZIP.") from exc
    if not {"pyproject.toml", "LICENSE", "README.md"}.issubset(paths):
        raise ReleaseError("Notebook payload is missing package metadata or licenses.")
    return paths


def audit_notebook(content: bytes, root: Path, *, compare_payload: bool = True) -> None:
    try:
        notebook = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReleaseError("Notebook is not valid UTF-8 JSON.") from exc
    if not isinstance(notebook, dict) or set(notebook) - {"cells", "metadata", "nbformat", "nbformat_minor"}:
        raise ReleaseError("Notebook contains unreviewed top-level data.")
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ReleaseError("Notebook must be in notebook format 4.")
    if not all(isinstance(cell, dict) for cell in notebook["cells"]):
        raise ReleaseError("Notebook contains an invalid cell.")
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ReleaseError("Invalid notebook metadata.")
    allowed_metadata = {"kernelspec", "language_info", "colab", "videocolor_payload_sha256"}
    if set(metadata) - allowed_metadata:
        raise ReleaseError("Notebook contains unreviewed metadata keys.")
    colab_metadata = metadata.get("colab", {})
    if not isinstance(colab_metadata, dict) or set(colab_metadata) - {"provenance"}:
        raise ReleaseError("Notebook contains unreviewed Colab metadata.")
    if colab_metadata.get("provenance", []):
        raise ReleaseError("Notebook contains Colab execution provenance.")
    encoded = _embedded_string(notebook["cells"])
    try:
        payload = base64.b64decode("".join(encoded.split()), validate=True)
    except ValueError as exc:
        raise ReleaseError("Notebook source payload is not valid base64.") from exc
    if metadata.get("videocolor_payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise ReleaseError("Notebook source payload hash does not match its metadata.")
    for cell in notebook["cells"]:
        if set(cell) - {"id", "cell_type", "metadata", "source", "outputs", "execution_count", "attachments"}:
            raise ReleaseError("Notebook cell contains unreviewed data.")
        if cell.get("cell_type") not in {"code", "markdown"}:
            raise ReleaseError("Unsupported notebook cell type.")
        if cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments"):
            raise ReleaseError("Notebook includes execution outputs, counts, or attachments.")
        cell_metadata = cell.get("metadata", {})
        if not isinstance(cell_metadata, dict) or set(cell_metadata) - {"id", "cellView", "tags"}:
            raise ReleaseError("Notebook cell contains unreviewed metadata.")
        audit_text(_source(cell).replace(encoded, "<embedded-source>"), NOTEBOOK)
        audit_text(json.dumps(cell_metadata), "notebook cell metadata")
        audit_text(json.dumps(cell.get("id", "")), "notebook cell ID")
    audit_text(json.dumps(metadata), "notebook metadata")
    embedded_contents = audit_payload(payload)
    if compare_payload:
        builder_path = root / "scripts" / "build_notebook.py"
        if not builder_path.is_file() or builder_path.is_symlink():
            raise ReleaseError("Notebook builder is missing or unsafe.")
        spec = importlib.util.spec_from_file_location("videocolor_notebook_builder", builder_path)
        if spec is None or spec.loader is None:
            raise ReleaseError("Cannot load notebook builder.")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        if audit_payload(builder.payload_bytes()) != embedded_contents:
            raise ReleaseError("Notebook contains stale source. Run python scripts/build_notebook.py first.")


def audited_contents(root: Path = ROOT) -> dict[str, bytes]:
    """Return the exact audited bytes so the packager does not reread files."""
    root = root.resolve()
    contents: dict[str, bytes] = {}
    for path in public_files(root):
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_TEXT_BYTES:
            raise ReleaseError(f"Source file is unexpectedly large: {relative}")
        content = path.read_bytes()
        if relative == NOTEBOOK:
            audit_notebook(content, root)
        else:
            try:
                audit_text(content.decode("utf-8"), relative)
            except UnicodeDecodeError as exc:
                raise ReleaseError(f"Source file must be UTF-8 text: {relative}") from exc
        contents[relative] = content
    required = {"README.md", "LICENSE", "DISCLAIMER.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".gitignore", NOTEBOOK,
                "src/videocolor/__init__.py", "src/videocolor/cli.py", "src/videocolor/_vendor/ddcolor/NOTICE.md"}
    required.update(f"src/videocolor/_vendor/ddcolor/LICENSE-{name}.txt" for name in
                    ("DDColor", "ConvNeXt", "Mask2Former", "DETR", "FastAI", "DeOldify"))
    if missing := required - contents.keys():
        raise ReleaseError("Required release files are missing: " + ", ".join(sorted(missing)))
    return contents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        contents = audited_contents()
    except (ReleaseError, OSError, ValueError) as exc:
        print(f"Release check failed: {exc}", file=sys.stderr)
        return 1
    print(f"Release source check passed: {len(contents)} files; notebook outputs and embedded source checked.")
    print("Private working files, videos, outputs, and model caches are outside the release selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a checked public-source ZIP; never archive the working folder wholesale."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

from check_release import ROOT, ReleaseError, audited_contents


def package_release(output: Path, root: Path = ROOT) -> list[str]:
    """Create a new archive from audited bytes, refusing to replace any file."""
    output = output.expanduser().absolute()
    if output.suffix.lower() != ".zip":
        raise ReleaseError("The release destination must end with .zip.")
    if output.exists() or output.is_symlink():
        raise ReleaseError("Release destination already exists. Choose a new --output path.")
    if not output.parent.is_dir():
        raise ReleaseError("The release destination's parent directory must exist.")
    contents = audited_contents(root)
    with output.open("xb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(contents.items()):
                entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, content)
    return sorted(contents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "VideoColor-v0.3.0-source.zip",
                        help="New ZIP destination (an existing file is never overwritten)")
    args = parser.parse_args(argv)
    try:
        files = package_release(args.output)
    except (ReleaseError, OSError, ValueError) as exc:
        print(f"Release packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"Created {args.output.absolute()} ({len(files)} source files)")
    for name in files:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

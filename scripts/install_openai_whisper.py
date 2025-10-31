#!/usr/bin/env python3
"""Install the Whisper dependency."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/openai/whisper.git"
PACKAGE_NAME = "openai-whisper"


class InstallError(RuntimeError):
    """Raised when the installation fails."""


def clone(destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--filter=blob:none", REPO_URL, str(destination)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def install_package(source_dir: Path, *, python: Path) -> None:
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(source_dir)],
        check=True,
    )


def install_whisper(*, python: Path, force: bool = False) -> None:
    """Install the Whisper package if needed."""
    result = subprocess.run(
        [str(python), "-m", "pip", "show", PACKAGE_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0 and not force:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "whisper"
        clone(repo_dir)
        install_package(repo_dir, python=python)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Reinstall even if already present.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Path to the Python interpreter whose pip should be used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    python_path = Path(args.python).resolve()
    if not python_path.exists():
        raise InstallError(f"Python interpreter not found: {python_path}")

    try:
        install_whisper(python=python_path, force=args.force)
    except subprocess.CalledProcessError as exc:
        raise InstallError("Failed to install patched Whisper dependency") from exc

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

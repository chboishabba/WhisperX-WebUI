#!/usr/bin/env python3
"""Install the Whisper dependency with Python 3.13 compatibility."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/jhj0517/jhj0517-whisper.git"
REPO_COMMIT = "197244318d5d75d9d195bff0705ab05a591684ec"
PACKAGE_NAME = "openai-whisper"


class InstallError(RuntimeError):
    """Raised when the patched installation fails."""


def patch_setup_file(setup_path: Path) -> None:
    """Patch the repository's setup.py to work on Python >= 3.13."""
    original = '    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))\n    return locals()["__version__"]'
    replacement = (
        '    namespace = {}\n'
        '    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"), namespace)\n'
        '    return namespace["__version__"]'
    )

    text = setup_path.read_text()
    if original not in text:
        raise InstallError("Unexpected setup.py contents; cannot apply Whisper patch.")

    setup_path.write_text(text.replace(original, replacement))


def clone_and_patch(destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--filter=blob:none", REPO_URL, str(destination)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run([
        "git",
        "-C",
        str(destination),
        "checkout",
        REPO_COMMIT,
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    patch_setup_file(destination / "setup.py")


def install_package(source_dir: Path, *, python: Path) -> None:
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(source_dir)],
        check=True,
    )


def install_whisper(*, python: Path, force: bool = False) -> None:
    """Install the patched Whisper package if needed."""
    result = subprocess.run(
        [str(python), "-m", "pip", "show", PACKAGE_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0 and not force:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "whisper"
        clone_and_patch(repo_dir)
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

#!/usr/bin/env python3
"""
Cross-platform bootstrap script for the development environment.
Works on Windows, macOS, and Linux — replaces setup.sh and setup.bat.

Usage:
    python bootstrap_env.py              # Create or update environment
    python bootstrap_env.py --recreate   # Remove and recreate from scratch
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ENV_FILE = "environment.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    *,
    stream: bool = False,
) -> tuple[bool, str]:
    """Run *cmd* and return ``(ok, stdout)``.

    When *stream* is True stdout/stderr are forwarded to the terminal in real
    time (useful for long-running installs).  Otherwise output is captured and
    returned as a string.
    """
    try:
        if stream:
            subprocess.run(cmd, check=True)
            return True, ""
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, getattr(e, "stderr", "") or ""
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"


def conda_run(env: str, *cmd: str, stream: bool = False) -> tuple[bool, str]:
    """Run a command inside the given conda environment."""
    return run(
        ["conda", "run", "--no-capture-output", "-n", env, *cmd],
        stream=stream,
    )


def get_env_name() -> str | None:
    """Parse the environment name from environment.yml (no YAML dep needed)."""
    path = Path(ENV_FILE)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.strip().startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def env_exists(name: str) -> bool:
    """Return True if a conda environment called *name* already exists."""
    ok, stdout = run(["conda", "env", "list"])
    if not ok:
        return False
    return any(
        parts[0] == name
        for line in stdout.splitlines()
        if (parts := line.split()) and not line.startswith("#")
    )


def header(msg: str) -> None:
    print(f"\n{'─' * 55}\n  {msg}\n{'─' * 55}")


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------


def check_conda() -> bool:
    header("Checking for conda")
    ok, stdout = run(["conda", "--version"])
    if not ok:
        print("  conda not found.")
        print("  Install Miniforge: https://github.com/conda-forge/miniforge")
        print("  Then restart your terminal and re-run this script.")
        return False
    print(f"  {stdout.strip()}")
    return True


def check_git() -> bool:
    header("Checking for git")
    ok, stdout = run(["git", "--version"])
    if not ok:
        print("  git not found — pre-commit hooks will be skipped.")
        return False
    ok, _ = run(["git", "rev-parse", "--git-dir"])
    if not ok:
        print("  Not inside a git repository — pre-commit hooks will be skipped.")
        return False
    print(f"  {stdout.strip()}")
    return True


def setup_environment(name: str, *, recreate: bool = False) -> bool:
    header(f"Setting up conda environment '{name}'")
    exists = env_exists(name)

    if exists and recreate:
        print("  Removing existing environment ...")
        ok, _ = run(["conda", "env", "remove", "-n", name, "-y"], stream=True)
        if not ok:
            print("  Failed to remove environment.")
            return False
        exists = False

    if exists:
        print("  Environment exists — updating ...")
        ok, _ = run(["conda", "env", "update", "-f", ENV_FILE, "--prune"], stream=True)
    else:
        print("  Creating environment ...")
        ok, _ = run(["conda", "env", "create", "-f", ENV_FILE], stream=True)

    if not ok:
        print("  Failed to set up environment.")
        if not recreate:
            print("  Tip: re-run with --recreate to start from scratch.")
        return False

    print("  Done")
    return True


def install_dev_tools(name: str) -> bool:
    header("Installing dev tools (ruff, mypy, pytest, pre-commit)")
    ok, _ = conda_run(name, "pip", "install", ".[dev]", stream=True)
    if not ok:
        print("  Failed to install dev tools.")
        return False
    print("  Done")
    return True


def install_pre_commit_hooks(name: str) -> bool:
    header("Installing pre-commit hooks")
    ok, _ = conda_run(name, "pre-commit", "install")
    if not ok:
        print("  Failed — you can install them manually later:")
        print(f"    conda activate {name}")
        print("    pre-commit install")
        return False
    print("  Done")
    return True


def verify(name: str) -> bool:
    header("Verifying installation")
    tools = {
        "python": ["python", "--version"],
        "ruff": ["ruff", "--version"],
        "mypy": ["mypy", "--version"],
        "pytest": ["pytest", "--version"],
        "pre-commit": ["pre-commit", "--version"],
    }
    all_ok = True
    for label, cmd in tools.items():
        ok, stdout = conda_run(name, *cmd)
        if ok:
            version = stdout.strip().splitlines()[0] if stdout.strip() else "ok"
            print(f"  [ok]   {label:12s} {version}")
        else:
            print(f"  [FAIL] {label:12s} not found")
            all_ok = False

    ok, _ = conda_run(name, "python", "src/example.py")
    if ok:
        print(f"  [ok]   {'example':12s} src/example.py ran successfully")
    else:
        print(f"  [FAIL] {'example':12s} src/example.py failed")
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the development environment (cross-platform).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Remove and recreate the conda environment from scratch.",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Development Environment Bootstrap")
    print("=" * 55)
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]}")

    if not Path(ENV_FILE).exists():
        print(f"\n  ERROR: {ENV_FILE} not found.")
        print("  Run this script from the project root directory.")
        return 1

    env_name = get_env_name()
    if not env_name:
        print(f"\n  ERROR: could not read environment name from {ENV_FILE}.")
        return 1
    print(f"  Env name : {env_name}")

    # Prerequisites
    if not check_conda():
        return 1
    has_git = check_git()

    # Environment
    if not setup_environment(env_name, recreate=args.recreate):
        return 1
    if not install_dev_tools(env_name):
        return 1

    # Pre-commit (best-effort — only if git is available)
    if has_git:
        install_pre_commit_hooks(env_name)

    # Verify
    verify(env_name)

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("=" * 55)
    print(
        f"""
Next steps:
  1. Activate the environment:
       conda activate {env_name}
  2. (VS Code users) Configure your editor:
       - Install recommended extensions when prompted
       - Copy .vscode/settings_template.json to .vscode/settings.json
  3. Start coding!

Useful commands:
  ruff check .                 Lint
  ruff check --fix .           Lint + auto-fix
  ruff format .                Format code
  mypy .                       Type-check
  pytest                       Run tests
  pre-commit run --all-files   Run all pre-commit hooks
"""
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted.")
        sys.exit(130)

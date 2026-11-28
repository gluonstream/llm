"""Fetch, build, and install the latest llm from GitHub."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

from . import __version__, host

REPO_URL = "https://github.com/gluonstream/llm.git"
LAST_VERSION_URL = (
    "https://raw.githubusercontent.com/gluonstream/llm/"
    "main/LAST_VERSION.md")
CHECKOUT_NAME = "llm-update"
BRANCH = "main"
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class UpdateError(RuntimeError):
    pass


def temp_root(platform_name=None):
    name = sys.platform if platform_name is None else platform_name
    return Path("/private/tmp" if name == "darwin" else "/tmp")


def default_checkout(root=None):
    configured = os.environ.get("LMM_UPDATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return (Path(root) if root is not None else temp_root()) / CHECKOUT_NAME


def _version_tuple(value):
    match = _VERSION_RE.search(str(value or ""))
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def latest_version(url=LAST_VERSION_URL, opener=urllib.request.urlopen,
                   timeout=10):
    """Return the version published in LAST_VERSION.md, or None."""
    try:
        with opener(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    match = _VERSION_RE.search(text)
    return ".".join(match.groups()) if match else None


def needs_update(latest, current=__version__):
    remote = _version_tuple(latest)
    have = _version_tuple(current)
    if remote is None or have is None:
        return False
    return remote > have


def _run(command, cwd=None, runner=subprocess.run):
    try:
        runner(command, cwd=str(cwd) if cwd is not None else None, check=True)
    except subprocess.CalledProcessError as exc:
        raise UpdateError("command failed: %s" % " ".join(command)) from exc
    except OSError as exc:
        raise UpdateError("command failed: %s: %s"
                          % (" ".join(command), exc)) from exc


def update(checkout=None, repo_url=REPO_URL, runner=subprocess.run):
    if not (host.IS_MACOS or host.IS_LINUX):
        raise UpdateError("unsupported OS: %s (supported: macOS, Arch Linux)"
                          % sys.platform)
    if shutil.which("git") is None:
        raise UpdateError("git not found; install git and retry")
    checkout = Path(checkout).expanduser() if checkout is not None else default_checkout()
    if checkout.exists():
        if not (checkout / ".git").is_dir():
            raise UpdateError("%s is not a git checkout; remove it and retry" % checkout)
        _run(["git", "-C", str(checkout), "fetch", "origin", BRANCH],
             runner=runner)
        _run(["git", "-C", str(checkout), "checkout", "-f", "-B", BRANCH,
              "origin/" + BRANCH], runner=runner)
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", BRANCH, repo_url, str(checkout)],
             runner=runner)
    _run(["./build.sh"], cwd=checkout, runner=runner)
    _run(["./install.sh"], cwd=checkout, runner=runner)
    return checkout


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llm update",
        description="Fetch the latest source, build it, and install it.")
    parser.add_argument("--dir", help="checkout directory")
    parser.add_argument("--force", action="store_true",
                        help="update even when LAST_VERSION.md is not newer")
    args = parser.parse_args(argv)
    if not args.force:
        latest = latest_version()
        if latest is not None and not needs_update(latest, __version__):
            print("llm %s is up to date (latest %s)." % (__version__, latest))
            return 0
    try:
        checkout = update(checkout=args.dir)
    except UpdateError as exc:
        print("llm update: %s" % exc, file=sys.stderr)
        return 1
    print("Updated llm from %s" % checkout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

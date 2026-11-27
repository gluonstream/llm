"""Fetch, build, and install the latest llm from GitHub."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import host

REPO_URL = "https://github.com/gluonstream/llm.git"
CHECKOUT_NAME = "llm-update"


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
        _run(["git", "-C", str(checkout), "pull", "--ff-only"], runner=runner)
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(checkout)], runner=runner)
    _run(["./build.sh"], cwd=checkout, runner=runner)
    _run(["./install.sh"], cwd=checkout, runner=runner)
    return checkout


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llm update",
        description="Fetch the latest source, build it, and install it.")
    parser.add_argument("--dir", help="checkout directory")
    args = parser.parse_args(argv)
    try:
        checkout = update(checkout=args.dir)
    except UpdateError as exc:
        print("llm update: %s" % exc, file=sys.stderr)
        return 1
    print("Updated llm from %s" % checkout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

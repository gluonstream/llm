"""Diagnose llama.cpp availability and model load failures."""
import re
import subprocess

from . import config, host

ARCH_RE = re.compile(r"unknown model architecture: '([^']+)'")
VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
BUILD_RE = re.compile(r"build\s+(\d+)")
CONTEXT_OVERFLOW_RE = re.compile(
    r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")


def parse_version(text):
    match = VERSION_RE.search(text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def version_at_least(installed, required):
    have = parse_version(installed)
    need = parse_version(required)
    if have is None or need is None:
        return False
    return have >= need


def explain_failure(log_text, upgrade_hint=None):
    """Turn the tail of server.log into one actionable line, or None."""
    hint = upgrade_hint or host.llama_cpp_upgrade_hint()
    text = log_text or ""
    arch = ARCH_RE.search(text)
    if arch:
        name = arch.group(1)
        if name.lower().startswith("inkling2"):
            return ("unknown model architecture '%s' \u2192 Inkling 2 needs llama.cpp >= 1.1.0; "
                    "run: %s" % (name, hint))
        return ("unknown model architecture '%s' \u2192 upgrade your llama.cpp build; "
                "run: %s" % (name, hint))
    overflow = None
    for line in text.splitlines():
        match = CONTEXT_OVERFLOW_RE.search(line)
        if match:
            overflow = match
    if overflow:
        requested = int(overflow.group(1))
        available = int(overflow.group(2))
        return ("request of %s tokens exceeds the %s-token context \u2192 "
                "run `llm apply` (or `llm context <model> N`) and restart"
                % ("{:,}".format(requested), "{:,}".format(available)))
    failure = None
    for line in text.splitlines():
        if "failed to load model" in line:
            failure = line.strip()
    if failure:
        return "%s \u2192 see `lmm logs`" % failure
    return None


def llama_server_version(llama_server=None):
    result = {"raw": "", "version": None, "build": None}
    try:
        completed = subprocess.run(
            [llama_server or config.LLAMA_SERVER, "--version"], capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return result
    raw = (completed.stdout or completed.stderr or "").strip()
    result["raw"] = raw.splitlines()[0] if raw else ""
    version = parse_version(raw)
    if version:
        result["version"] = ".".join(str(part) for part in version)
    build = BUILD_RE.search(raw)
    if build:
        result["build"] = int(build.group(1))
    return result


def brew_llama_cpp_status():
    """Return a one-line Homebrew status, or None when brew is unavailable."""
    try:
        completed = subprocess.run(
            ["brew", "info", "llama.cpp"], capture_output=True, text=True,
            check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (completed.stdout or "") + (completed.stderr or "")
    installed = None
    available = None
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"llama\.cpp\s+(\d+\.\d+\.\d+)\s+\u2192\s+(?:stable\s+)?(\d+\.\d+\.\d+)",
                         stripped)
        if match:
            installed, available = match.group(1), match.group(2)
            break
        match = re.match(r"==> llama\.cpp:\s+(\d+\.\d+\.\d+)\s+\u2192", stripped)
        if match:
            installed = match.group(1)
    if available and installed and available != installed:
        return "Homebrew has %s; installed %s (run: brew upgrade llama.cpp)" % (
            available, installed)
    if installed:
        return "Homebrew llama.cpp %s" % installed
    return None


def llama_cpp_package_status():
    if host.IS_MACOS:
        return brew_llama_cpp_status()
    if not host.IS_LINUX:
        return None
    path = config.LLAMA_SERVER
    try:
        completed = subprocess.run(
            ["pacman", "-Qo", path], capture_output=True, text=True,
            check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    text = "%s%s" % (completed.stdout or "", completed.stderr or "")
    match = re.search(r"is owned by (\S+) (\S+)", text)
    if match:
        return "Arch package %s %s" % (match.group(1), match.group(2))
    return None

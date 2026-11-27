"""Manage the persistent llama.cpp server used by llm."""
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from . import config
from .config import PORT, URL, selected_profile
from .diagnose import explain_failure

PIDFILE = config.STATE / "process.json"
LOG = config.STATE / "server.log"
LOG_MAX_BYTES = 20 * 1024 * 1024

HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ModelLoadError(RuntimeError):
    pass


@contextmanager
def locked():
    """Serialize start/stop/select mutations across processes."""
    config.STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.STATE / "lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def identity(pid):
    try:
        result = subprocess.run(
            [config.PS, "-p", str(pid), "-o", "lstart=", "-o", "command="],
            capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def uptime(pid):
    try:
        result = subprocess.run(
            [config.PS, "-p", str(pid), "-o", "etime="],
            capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def running():
    if not PIDFILE.exists():
        return None
    try:
        record = json.loads(PIDFILE.read_text())
        pid = record["pid"]
        if type(pid) is not int or pid <= 0:
            return None
        actual = identity(pid)
        if actual and actual == record["identity"] and "llama-server" in actual:
            if any(profile["alias"] in actual for profile in config.PROFILES.values()):
                return pid
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def healthy(timeout=2):
    try:
        with HTTP.open(URL + "/health", timeout=timeout) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError):
        return False


def listener_pid():
    try:
        result = subprocess.run(
            [config.LSOF, "-nP", "-tiTCP:%d" % PORT, "-sTCP:LISTEN"],
            capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    return None


def _source(profile):
    return profile.get("path") or profile.get("hf") or ""


def is_expected_listener(pid, profile=None):
    actual = identity(pid) if pid else ""
    profile = profile or selected_profile()
    source = _source(profile)
    return ("llama-server" in actual and profile["alias"] in actual
            and source in actual and "--port %d" % PORT in actual
            and "--host 127.0.0.1" in actual)


def server_args(profile, llama_server=None):
    args = [llama_server or config.LLAMA_SERVER]
    if "path" in profile:
        args += ["-m", profile["path"], "--offline", "-lm", "mmap", "--lazy-mode", "on"]
    else:
        args += ["-hf", profile["hf"]]
    args += ["--no-mmproj", "-ngl", "99", "-fa", "on",
             "-c", str(profile["context"]), "-b", "512", "-ub", "256",
             "--parallel", "1", "--host", "127.0.0.1", "--port", str(PORT),
             "--alias", profile["alias"], "--jinja", "--no-webui"]
    if profile["id"] == "qwen2.5-7b-instruct-q4-k-m":
        args += ["--reasoning", "off"]
    sampling = profile["sampling"]
    args += ["--temp", str(sampling["temperature"]), "--top-p", str(sampling["top_p"]),
             "--min-p", str(sampling["min_p"]),
             "--repeat-penalty", str(sampling["repeat_penalty"])]
    return args


def validate_profile_files(profile):
    path = profile.get("path")
    if not path:
        return
    model = Path(path)
    try:
        size = model.stat().st_size
    except OSError:
        raise ModelLoadError(
            "Model file missing: %s — run `lmm doctor` or download it again" % model)
    if size <= 0:
        raise ModelLoadError(
            "Model file is empty: %s — download it again" % model)


def tail_text(path, max_bytes=64 * 1024):
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def rotate_log(path=None, max_bytes=LOG_MAX_BYTES):
    path = Path(path or LOG)
    if path.name.endswith(".1"):
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < max_bytes:
        return False
    target = path.with_name(path.name + ".1")
    try:
        os.replace(path, target)
    except OSError:
        return False
    return True


def last_failure(path=None):
    return explain_failure(tail_text(path or LOG))


def log_size(path=None):
    try:
        return Path(path or LOG).stat().st_size
    except OSError:
        return 0


def wait_until_port_bindable(timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", PORT))
            return
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def stop():
    pid = running()
    if pid:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while running() and time.monotonic() < deadline:
            time.sleep(0.2)
        if running():
            os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while running() and time.monotonic() < deadline:
                time.sleep(0.2)
        if running():
            raise RuntimeError("Could not stop PID %s" % pid)
        port_deadline = time.monotonic() + 10
        while listener_pid() and time.monotonic() < port_deadline:
            time.sleep(0.2)
        remaining = listener_pid()
        if remaining:
            raise RuntimeError(
                "Port %d is still held by PID %s after stopping PID %s"
                % (PORT, remaining, pid))
    PIDFILE.unlink(missing_ok=True)


def start(expected=None):
    profile = selected_profile()
    if expected is not None:
        wanted = config.resolve_model(expected)["id"]
        if wanted != profile["id"]:
            raise ModelLoadError("Selected model changed; rerun the label command")
    validate_profile_files(profile)
    pid = running()
    if pid and not is_expected_listener(pid, profile):
        raise ModelLoadError("Managed server differs from selection; run `lmm restart`")
    created = False
    if not pid:
        existing_listener = listener_pid()
        if existing_listener:
            if not is_expected_listener(existing_listener, profile):
                raise ModelLoadError(
                    "127.0.0.1:%d is already held by PID %s; it is not the configured "
                    "llm server" % (PORT, existing_listener))
            pid = existing_listener
            PIDFILE.write_text(json.dumps({"pid": pid, "identity": identity(pid)}))
        else:
            wait_until_port_bindable()
        args = server_args(profile)
        if not pid:
            config.STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
            rotate_log(LOG)
            with LOG.open("ab") as log:
                process = subprocess.Popen(
                    args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True)
            pid = process.pid
            PIDFILE.write_text(json.dumps({"pid": pid, "identity": identity(pid)}))
            created = True
    return pid, created, profile


def wait_ready(pid, created, profile, timeout=3600, poll=1.0, on_loading=None):
    began = time.monotonic()
    last_bucket = -1
    try:
        while True:
            if running() != pid:
                message = last_failure()
                raise ModelLoadError(
                    message or "Model server exited; see `lmm logs` (%s)" % LOG)
            if healthy():
                return
            elapsed = time.monotonic() - began
            if elapsed >= timeout:
                raise ModelLoadError(
                    "Model startup exceeded %d minutes; see `lmm logs` (%s)"
                    % (timeout // 60, LOG))
            bucket = int(elapsed) // 15
            if bucket != last_bucket:
                last_bucket = bucket
                if on_loading is not None:
                    on_loading(elapsed)
                elif elapsed >= 1:
                    print("Loading model: %ds" % elapsed, flush=True)
            time.sleep(poll)
    except BaseException:
        if created:
            with (config.STATE / "lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if running() == pid:
                    stop()
        raise

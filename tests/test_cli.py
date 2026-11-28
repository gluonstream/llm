import contextlib
import hashlib
import io
import json
import os
import socketserver
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import cli, config, hardware, hub, runtime, service


def run_cli(argv, locked=None):
    if locked is None:
        locked = lambda: contextlib.nullcontext()
    with mock.patch.object(config, "migrate_state", return_value=False):
        with mock.patch.object(service, "locked", locked):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return cli.main(argv)


class CliTest(unittest.TestCase):
    def _main(self, argv, locked=None):
        return run_cli(argv, locked)

    def test_help_exits_zero_and_names_commands(self):
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False):
            with redirect_stdout(buffer):
                self.assertEqual(cli.main(["help"]), 0)
        text = buffer.getvalue()
        for command in ("list", "status", "use", "doctor", "logs", "version"):
            self.assertIn(command, text)

    def test_prog_follows_invocation_name(self):
        with mock.patch.object(sys, "argv", ["/Users/x/.local/bin/llm"]):
            self.assertEqual(cli.build_parser().prog, "llm")
        with mock.patch.object(sys, "argv", ["/Users/x/.local/bin/lmm"]):
            self.assertEqual(cli.build_parser().prog, "lmm")

    def test_select_updates_selection_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selected-model.json"
            with mock.patch.object(config, "SELECTION", selection):
                self.assertEqual(self._main(["select", "inkling"]), 0)
            self.assertEqual(config.selected_profile(selection=selection)["id"],
                             "inkling-2-mini-gguf-ud-q4-k-xl")

    def test_unknown_model_reports_choices(self):
        with mock.patch.object(config, "SELECTION", Path("/nonexistent/selection.json")):
            self.assertEqual(self._main(["use", "nope"]), 1)

    def test_status_warns_when_server_context_differs(self):
        selected = {"id": "qwen2.5-7b-instruct-q4-k-m", "context": 65536}
        running = {"id": "qwen2.5-7b-instruct-q4-k-m", "ready": True,
                   "pid": 42, "uptime": "4m"}
        buffer = io.StringIO()
        with mock.patch.object(config, "selected_profile", return_value=selected), \
             mock.patch.object(cli, "_running_info", return_value=running), \
             mock.patch.object(service, "server_context", return_value=8192), \
             mock.patch.object(config, "migrate_state", return_value=False), \
             redirect_stdout(buffer):
            self.assertEqual(cli.main(["status"]), 0)
        self.assertIn("warning", buffer.getvalue())


class CliFixTest(unittest.TestCase):
    def test_local_size_missing_multishard_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            self.assertIsNone(cli._local_size({"path": str(path)}))

    def test_local_size_zero_byte_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.touch()
            self.assertIsNone(cli._local_size({"path": str(path)}))

    def test_local_size_sums_only_the_shard_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = directory / "model-Q4_K_M-00001-of-00002.gguf"
            second = directory / "model-Q4_K_M-00002-of-00002.gguf"
            other = directory / "model-Q8_0.gguf"
            first.write_bytes(b"x" * 10)
            second.write_bytes(b"x" * 20)
            other.write_bytes(b"x" * 100)
            self.assertEqual(
                cli._local_size({"path": str(first)}), 30)

    def test_select_stops_running_service_when_switching(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selected-model.json"
            config.select_profile("inkling", selection=selection)
            with mock.patch.object(config, "SELECTION", selection), \
                 mock.patch.object(service, "running", return_value=123), \
                 mock.patch.object(service, "stop") as stop:
                self.assertEqual(run_cli(["select", "glm-4.7-flash-ud-q4-k-xl"]), 0)
                stop.assert_called_once()

    def test_start_uses_service_lock(self):
        entered = []

        @contextlib.contextmanager
        def fake_locked():
            entered.append(True)
            yield

        profile = config.PROFILES[config.DEFAULT]
        with mock.patch.object(service, "start", return_value=(1, False, profile)), \
             mock.patch.object(service, "wait_ready"):
            self.assertEqual(run_cli(["start"], locked=fake_locked), 0)
        self.assertTrue(entered)


class HubCliTest(unittest.TestCase):
    def test_search_prints_repo(self):
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(hub, "search_models",
                               return_value=[{"id": "unsloth/Inkling-2-Mini-GGUF",
                                              "downloads": 10, "likes": 2,
                                              "gated": False, "tags": ["gguf"]}]), \
             redirect_stdout(buffer):
            self.assertEqual(cli.main(["search", "inkling"]), 0)
        self.assertIn("unsloth/Inkling-2-Mini-GGUF", buffer.getvalue())

    def test_registered_model_matches_full_relative_filename(self):
        profiles = {"a": {"repo": "o/r", "path": "/m/a/M.gguf",
                          "filename": "a/M.gguf"}}
        self.assertIsNone(cli._registered_model(profiles, "o/r", "b/M.gguf"))


class GetCacheTest(unittest.TestCase):
    """Exercise `get` end to end against a local HTTP server and temp cache."""

    COMMIT = "a" * 40
    PAYLOAD = b"gguf-bytes"
    FILENAME = "M-Q4.gguf"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # hfcache.cache_root() resolves the env path; match that here.
        self.cache = (self.root / "hub").resolve()
        self.state = self.root / "state" / "llm"
        self.legacy = self.root / "legacy-models"
        self.sha = hashlib.sha256(self.PAYLOAD).hexdigest()
        self.requests = []
        self.siblings = [self._sibling(self.FILENAME)]

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append(self.path)
                if self.path.startswith("/api/models/"):
                    body = json.dumps({
                        "sha": outer.COMMIT,
                        "siblings": outer.siblings,
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(outer.PAYLOAD)))
                self.end_headers()
                self.wfile.write(outer.PAYLOAD)

        class TestServer(HTTPServer):
            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                host, port = self.server_address[:2]
                self.server_name = host
                self.server_port = port

        self.server = TestServer(("127.0.0.1", 0), Handler)
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

        self._patch(mock.patch.dict(
            os.environ, {"HF_HUB_CACHE": str(self.cache)}))
        self._patch(mock.patch.object(hub, "BASE_URL", self.base))
        self._patch(mock.patch.object(config, "STATE", self.state))
        self._patch(mock.patch.object(config, "USER_MODELS",
                                      self.state / "models.json"))
        self._patch(mock.patch.object(config, "PROFILES", {}))
        self._patch(mock.patch.object(cli, "MODELS_DIR", self.legacy))
        self._patch(mock.patch.object(hardware, "preflight", return_value=[]))
        self._patch(mock.patch.object(config, "migrate_state",
                                      return_value=False))

    def _patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def _sibling(self, name):
        return {"rfilename": name,
                "lfs": {"size": len(self.PAYLOAD), "sha256": self.sha}}

    def _run(self, argv):
        return run_cli(argv)

    def _repo_dir(self):
        return self.cache / "models--owner--repo"

    def _snapshot(self, filename=None):
        return (self._repo_dir() / "snapshots" / self.COMMIT
                / (filename or self.FILENAME))

    def _registry(self):
        return json.loads((self.state / "models.json").read_text())

    def _add_existing(self):
        existing = {
            "id": "existing",
            "alias": "existing",
            "repo": "owner/repo",
            "filename": self.FILENAME,
            "revision": "main",
            "path": str(self._snapshot()),
            "context": 4096,
            "sampling": {"temperature": 0},
        }
        config.PROFILES["existing"] = existing
        return existing

    def test_get_writes_snapshot_and_profile(self):
        self.assertEqual(self._run(["get", "owner/repo:Q4"]), 0)
        snapshot = self._snapshot()
        blob = self._repo_dir() / "blobs" / self.sha
        self.assertTrue(snapshot.is_symlink())
        self.assertEqual(snapshot.resolve(), blob.resolve())
        self.assertEqual(snapshot.read_bytes(), self.PAYLOAD)
        self.assertFalse(os.path.isabs(os.readlink(snapshot)))
        self.assertEqual(
            (self._repo_dir() / "refs" / "main").read_text(), self.COMMIT)
        self.assertTrue(any("/api/models/owner/repo" in path
                            for path in self.requests))
        self.assertFalse(
            (self._repo_dir() / "download-manifest.json").exists())

        profile = next(iter(self._registry().values()))
        self.assertEqual(profile["repo"], "owner/repo")
        self.assertEqual(profile["filename"], self.FILENAME)
        self.assertEqual(profile["revision"], "main")
        self.assertEqual(profile["path"], str(snapshot))
        self.assertEqual(profile["context"], 8192)
        self.assertEqual(profile["sampling"], config.DEFAULT_SAMPLING)

        manifest = self.state / "manifests" / (profile["id"] + ".json")
        self.assertTrue(manifest.is_file())
        data = json.loads(manifest.read_text())
        self.assertEqual(data["repo"], "owner/repo")
        self.assertEqual(data["sha256"], self.sha)
        self.assertEqual(data["commit"], self.COMMIT)
        self.assertFalse((snapshot.parent / "download-manifest.json").exists())

    def test_get_never_writes_local_share(self):
        self.assertEqual(self._run(["get", "owner/repo:Q4"]), 0)
        self.assertFalse(self.legacy.exists())
        self.assertFalse((self.root / ".local").exists())
        self.assertFalse(
            (self._repo_dir() / "download-manifest.json").exists())

    def test_get_registers_nested_filename(self):
        self.siblings = [self._sibling("sub/M-Q4.gguf")]
        self.assertEqual(self._run(["get", "owner/repo:Q4"]), 0)
        snapshot = self._snapshot("sub/M-Q4.gguf")
        self.assertTrue(snapshot.is_symlink())
        self.assertEqual(snapshot.read_bytes(), self.PAYLOAD)
        profile = next(iter(self._registry().values()))
        self.assertEqual(profile["filename"], "sub/M-Q4.gguf")
        self.assertEqual(profile["path"], str(snapshot))

    def test_get_ambiguous_selector_lists_candidates(self):
        files = [{"name": "A-Q4.gguf", "size": 1, "sha256": "a"},
                 {"name": "A-Q5.gguf", "size": 2, "sha256": "b"}]
        buffer = io.StringIO()
        with mock.patch.object(hub, "repo_info",
                               return_value=(self.COMMIT, files)), \
             redirect_stdout(io.StringIO()), redirect_stderr(buffer):
            self.assertEqual(cli.main(["get", "owner/repo"]), 1)
        self.assertIn("A-Q4.gguf", buffer.getvalue())
        self.assertIn("A-Q5.gguf", buffer.getvalue())

    def test_get_rejects_selector_plus_file(self):
        with mock.patch.object(hub, "repo_info") as info:
            self.assertEqual(
                self._run(["get", "owner/repo:Q4", "--file", "x.gguf"]), 1)
        info.assert_not_called()

    def test_get_is_idempotent_for_registered_model(self):
        existing = self._add_existing()
        self.assertEqual(self._run(["get", "owner/repo"]), 0)
        downloads = [path for path in self.requests if "/resolve/" in path]
        self.assertEqual(self._run(["get", "owner/repo"]), 0)
        self.assertEqual(
            [path for path in self.requests if "/resolve/" in path], downloads)
        profile = self._registry()["existing"]
        self.assertEqual(profile["id"], "existing")
        self.assertEqual(profile["path"], str(self._snapshot()))
        self.assertEqual(profile["context"], existing["context"])
        self.assertEqual(profile["sampling"], existing["sampling"])
        self.assertTrue((self._repo_dir() / "blobs" / self.sha).is_file())
        self.assertEqual(
            len(list((self._repo_dir() / "blobs").iterdir())), 1)

    def test_get_refuses_when_hardware_too_small(self):
        with mock.patch.object(hardware, "preflight",
                               return_value=["model does not fit"]), \
             mock.patch.object(hub, "download_blob") as download:
            self.assertEqual(self._run(["get", "owner/repo:Q4"]), 1)
        download.assert_not_called()
        self.assertFalse(self._repo_dir().exists())

    def test_get_force_downloads_when_hardware_refuses(self):
        with mock.patch.object(hardware, "preflight",
                               return_value=["model does not fit"]):
            self.assertEqual(self._run(["get", "owner/repo:Q4", "--force"]), 0)
        self.assertTrue((self._repo_dir() / "blobs" / self.sha).is_file())

    def test_get_existing_skips_preflight_without_force(self):
        self._add_existing()
        with mock.patch.object(hardware, "preflight") as preflight:
            self.assertEqual(self._run(["get", "owner/repo"]), 0)
        preflight.assert_not_called()

    def test_get_respects_context_override(self):
        self.assertEqual(
            self._run(["get", "owner/repo:Q4", "--context", "2048"]), 0)
        profile = next(iter(self._registry().values()))
        self.assertEqual(profile["context"], 2048)


class BuiltinGetTest(unittest.TestCase):
    """`get` on a built-in repo must not try to re-register the built-in."""

    COMMIT = "a" * 40

    def test_get_builtin_repo_does_not_register(self):
        builtin = config.BUILTIN_PROFILES["inkling-2-mini-gguf-ud-q4-k-xl"]
        repo = builtin["repo"]
        filename = builtin["filename"]
        payload = b"gguf-bytes"
        sha = hashlib.sha256(payload).hexdigest()
        chosen = {"name": filename, "size": len(payload), "sha256": sha}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = (root / "hub").resolve()
            state = root / "state" / "llm"
            blob = cache / "models--unsloth--Inkling-2-Mini-GGUF" / "blobs" / sha
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(payload)
            manifest = state / "manifests" / (builtin["id"] + ".json")

            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}), \
                 mock.patch.object(config, "STATE", state), \
                 mock.patch.object(config, "PROFILES",
                                   {builtin["id"]: dict(builtin)}), \
                 mock.patch.object(config, "migrate_state",
                                   return_value=False), \
                 mock.patch.object(hub, "repo_info",
                                   return_value=(self.COMMIT, [chosen])), \
                 mock.patch.object(hub, "select_file", return_value=chosen), \
                 mock.patch.object(hub, "download_blob", return_value=blob), \
                 mock.patch.object(hub, "sha256_file", return_value=sha), \
                 mock.patch.object(hub, "write_manifest",
                                   return_value=manifest), \
                 mock.patch.object(config, "register_model") as register, \
                 redirect_stdout(io.StringIO()) as out, \
                 redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["get", repo + ":Q4"]), 0)

            register.assert_not_called()
            self.assertIn("built-in", out.getvalue())
            self.assertTrue(blob.is_file())

    def test_list_prints_profile_warnings(self):
        warnings = ["dropped user model 'dup': alias 'q' is already used by x"]
        buffer = io.StringIO()
        with mock.patch.object(config, "PROFILE_WARNINGS", warnings), \
             mock.patch.object(config, "selected_profile",
                               return_value=config.PROFILES[config.DEFAULT]), \
             mock.patch.object(service, "running", return_value=None), \
             mock.patch.object(service, "last_failure", return_value=None), \
             mock.patch.object(config, "migrate_state", return_value=False), \
             redirect_stdout(io.StringIO()), redirect_stderr(buffer):
            self.assertEqual(cli.main(["list"]), 0)
        self.assertIn("warning: dropped user model", buffer.getvalue())


class MigrateCacheCliTest(unittest.TestCase):
    def test_migrate_cache_dry_run_prints_table(self):
        entry = {
            "source": "/tmp/graphify-qwen",
            "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "commit": "b" * 40,
            "revision": "main",
            "origin": "known",
            "status": "ready",
            "files": [{"filename": "qwen.gguf", "sha256": "a" * 64,
                       "size": 10, "path": "/tmp/graphify-qwen/qwen.gguf",
                       "present": True}],
        }
        result = {
            "source": entry["source"],
            "repo": entry["repo"],
            "commit": entry["commit"],
            "status": "dry-run",
            "files": [{"filename": "qwen.gguf", "status": "would migrate",
                       "method": None, "message": ""}],
            "leftover": [],
        }
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(cli.migrate, "plan", return_value=[entry]), \
             mock.patch.object(cli.migrate, "migrate",
                               return_value=result) as applied, \
             redirect_stdout(buffer):
            self.assertEqual(cli.main(["migrate-cache", "--dry-run"]), 0)
        text = buffer.getvalue()
        for value in ("Qwen/Qwen2.5-7B-Instruct-GGUF", "b" * 40,
                      "/tmp/graphify-qwen", "qwen.gguf"):
            self.assertIn(value, text)
        self.assertTrue(applied.call_args.kwargs["dry_run"])

    def test_migrate_cache_reports_missing(self):
        entry = {
            "source": "/tmp/graphify-gone",
            "repo": "org/gone",
            "commit": "c" * 40,
            "revision": "main",
            "origin": "known",
            "status": "missing",
            "files": [{"filename": "gone.gguf", "sha256": "a" * 64,
                       "size": 10, "path": "/tmp/graphify-gone/gone.gguf",
                       "present": False}],
        }
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(cli.migrate, "plan", return_value=[entry]), \
             mock.patch.object(cli.migrate, "cache_present",
                               return_value=False), \
             mock.patch.object(cli.migrate, "migrate") as applied, \
             redirect_stdout(buffer):
            self.assertEqual(cli.main(["migrate-cache"]), 0)
        text = buffer.getvalue()
        self.assertIn("org/gone", text)
        self.assertIn("missing", text)
        applied.assert_not_called()

    def test_migrate_cache_does_not_migrate_partial_split(self):
        payload = b"first shard only"
        sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "share"
            cache = root / "hub"
            source = (legacy / "graphify-split"
                      / "model-00001-of-00002.gguf")
            source.parent.mkdir(parents=True)
            source.write_bytes(payload)
            mapping = {
                "source": "~/.local/share/graphify-split",
                "repo": "org/split",
                "commit": "e" * 40,
                "revision": "main",
                "files": [
                    {"filename": "model-00001-of-00002.gguf",
                     "sha256": sha256, "size": len(payload)},
                    {"filename": "model-00002-of-00002.gguf",
                     "sha256": "f" * 64, "size": 5},
                ],
            }
            real_plan = cli.migrate.plan
            output = io.StringIO()
            with mock.patch.dict(os.environ,
                                 {"HF_HUB_CACHE": str(cache)}), \
                 mock.patch.object(cli.migrate, "KNOWN_MIGRATIONS",
                                   [mapping]), \
                 mock.patch.object(
                     cli.migrate, "plan",
                     side_effect=lambda: real_plan(legacy_root=legacy,
                                                   root=cache)), \
                 mock.patch.object(config, "migrate_state",
                                   return_value=False), \
                 redirect_stdout(output):
                self.assertEqual(cli.main(["migrate-cache"]), 0)
            text = output.getvalue()
            self.assertIn("missing", text)
            self.assertTrue(source.exists())
            self.assertFalse((cache / "models--org--split").exists())

    def test_migrate_cache_rerun_reports_already_present(self):
        payload = b"already migrated bytes"
        sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "share"
            cache = root / "hub"
            source = legacy / "graphify-test" / "model.gguf"
            source.parent.mkdir(parents=True)
            source.write_bytes(payload)
            mapping = {
                "source": "~/.local/share/graphify-test",
                "repo": "org/test",
                "commit": "d" * 40,
                "revision": "main",
                "files": [{"filename": "model.gguf", "sha256": sha256,
                           "size": len(payload)}],
            }
            real_plan = cli.migrate.plan

            def run():
                output = io.StringIO()
                with mock.patch.dict(os.environ,
                                     {"HF_HUB_CACHE": str(cache)}), \
                     mock.patch.object(cli.migrate, "KNOWN_MIGRATIONS",
                                       [mapping]), \
                     mock.patch.object(
                         cli.migrate, "plan",
                         side_effect=lambda: real_plan(legacy_root=legacy,
                                                       root=cache)), \
                     mock.patch.object(config, "migrate_state",
                                       return_value=False), \
                     redirect_stdout(output):
                    code = cli.main(["migrate-cache"])
                return code, output.getvalue()

            first_code, first = run()
            self.assertEqual(first_code, 0)
            self.assertIn("migrated", first)
            self.assertFalse(source.exists())

            second_code, second = run()
            self.assertEqual(second_code, 0)
            self.assertIn("already present", second)
            self.assertNotIn("missing", second)
            self.assertTrue(
                (cache / "models--org--test" / "blobs" / sha256).is_file())


class VersionCliTest(unittest.TestCase):
    def _output(self, argv):
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             redirect_stdout(buffer):
            self.assertEqual(cli.main(argv), 0)
        return buffer.getvalue()

    def test_version_prints_installed_and_up_to_date(self):
        self.assertEqual(self._output(["version"]).splitlines(), [
            "llm 1.2.0",
            "last version is 1.2.0. you are up to date.",
        ])

    def test_version_uses_invocation_name(self):
        with mock.patch.object(sys, "argv", ["/Users/x/.local/bin/lmm"]):
            lines = self._output(["version"]).splitlines()
        self.assertEqual(lines[0], "lmm 1.2.0")

    def test_version_points_to_update_when_latest_is_newer(self):
        with mock.patch("llm.LATEST_VERSION", "1.2.1"):
            lines = self._output(["version"]).splitlines()
        self.assertEqual(lines, [
            "llm 1.2.0",
            "last version is 1.2.1. run `llm update` to update.",
        ])


class DoctorCliTest(unittest.TestCase):
    GIB = 1024 ** 3

    def _machine(self, gpus=(), **kwargs):
        values = {
            "chip": "Apple M1", "model": "arm64",
            "total_ram": 16 * self.GIB, "unified": True, "gpu": "unknown",
            "vram": None, "disk_free": 100 * self.GIB,
            "available_ram": 8 * self.GIB,
        }
        values.update(kwargs)
        values["gpus"] = list(gpus)
        return hardware.Hardware(**values)

    def _doctor(self, machine=None, profiles=None, registry=None, pid=None,
                healthy=False, server_context=None, selected=None,
                opencode_mismatches=(), opencode_unknown=(),
                pi_mismatches=(), pi_unknown=(), codex_mismatches=(),
                linux=False, which=None):
        if machine is None:
            machine = self._machine()
        if registry is None:
            registry = config.Registry({}, {}, False, [])
        buffer = io.StringIO()
        patches = [
            mock.patch.object(config, "migrate_state", return_value=False),
            mock.patch.object(cli.diagnose, "llama_server_version",
                              return_value={"raw": "", "version": None,
                                            "build": None}),
            mock.patch.object(cli.diagnose, "llama_cpp_package_status",
                              return_value=None),
            mock.patch.object(cli.hardware, "detect_hardware",
                              return_value=machine),
            mock.patch.object(service, "running", return_value=pid),
            mock.patch.object(service, "healthy", return_value=healthy),
            mock.patch.object(service, "server_context",
                              return_value=server_context),
            mock.patch.object(service, "listener_pid", return_value=None),
            mock.patch.object(service, "last_failure", return_value=None),
            mock.patch.object(cli.opencode, "mismatches",
                              return_value=list(opencode_mismatches)),
            mock.patch.object(cli.opencode, "unknown_entries",
                              return_value=list(opencode_unknown)),
            mock.patch.object(cli.pi, "mismatches",
                              return_value=list(pi_mismatches)),
            mock.patch.object(cli.pi, "unknown_entries",
                              return_value=list(pi_unknown)),
            mock.patch.object(cli.codex, "mismatches",
                              return_value=list(codex_mismatches)),
            mock.patch.object(config, "load_registry", return_value=registry),
            mock.patch.object(config, "selected_profile",
                              return_value=selected),
            redirect_stdout(buffer),
        ]
        if profiles is not None:
            patches.append(mock.patch.object(config, "PROFILES", profiles))
        if linux:
            patches.append(mock.patch.object(cli.host, "IS_LINUX", True))
            patches.append(mock.patch.object(cli.shutil, "which",
                                             return_value=which))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            self.assertEqual(cli.main(["doctor"]), 0)
        return buffer.getvalue()

    def test_doctor_shows_context_and_gpu(self):
        gpu = hardware.GpuStats(
            name="NVIDIA RTX 4090", total=24 * self.GIB, used=3 * self.GIB,
            free=21 * self.GIB, utilization=35, source="gpustat")
        profiles = {"qwen": {"id": "qwen", "alias": "qwen",
                             "path": "/nonexistent/qwen.gguf",
                             "context": 65536, "sampling": {}}}
        text = self._doctor(machine=self._machine(gpus=[gpu]),
                            profiles=profiles)
        self.assertIn("64K", text)
        self.assertIn("35% util", text)
        self.assertIn("used", text)

    def test_doctor_warns_when_gpustat_missing(self):
        gpu = hardware.GpuStats(name="NVIDIA GeForce RTX 3090",
                                source="lspci")
        text = self._doctor(machine=self._machine(gpus=[gpu]),
                            linux=True, which=None)
        self.assertIn("pipx install gpustat", text)

    def test_doctor_warns_server_context_mismatch(self):
        selected = {"id": "qwen", "context": 65536}
        text = self._doctor(pid=42, healthy=True, server_context=8192,
                            selected=selected)
        self.assertIn("configured 64K", text)
        self.assertIn("server 8K", text)

    def test_doctor_warns_opencode_mismatch(self):
        text = self._doctor(opencode_mismatches=[("qwen", 65536, 8192)])
        self.assertIn("opencode.json", text)
        self.assertIn("--sync-opencode", text)

    def test_doctor_warns_pi_mismatch(self):
        text = self._doctor(pi_mismatches=[("qwen", 65536, 8192)])
        self.assertIn("pi models.json", text)
        self.assertIn("--sync-pi", text)

    def test_doctor_warns_codex_mismatch(self):
        selected = {"id": "qwen", "alias": "qwen", "context": 65536}
        text = self._doctor(selected=selected,
                            codex_mismatches=[("model", "gpt-5-codex", "qwen")])
        self.assertIn("codex config.toml", text)
        self.assertIn("--sync-codex", text)

    def test_doctor_shows_invalid_override_warning(self):
        registry = config.Registry({}, {}, False, [("x", "invalid context")])
        text = self._doctor(registry=registry)
        self.assertIn("warning: override for x ignored", text)

    def test_doctor_shows_cache_root_and_source(self):
        profiles = {
            "qwen2.5-7b-instruct-q4-k-m": {
                "id": "qwen2.5-7b-instruct-q4-k-m", "alias": "alpha",
                "path": "/nonexistent/a.gguf", "context": 8192,
                "sampling": {}},
            "user-model": {
                "id": "user-model", "alias": "beta",
                "path": "/nonexistent/b.gguf", "context": 8192,
                "sampling": {}},
            "cache-model": {
                "id": "cache-model", "alias": "gamma", "source": "cache",
                "path": "/nonexistent/c.gguf", "context": 8192,
                "sampling": {}},
        }
        registry = config.Registry({"user-model": profiles["user-model"]},
                                   {}, False, [])
        text = self._doctor(profiles=profiles, registry=registry)
        self.assertIn("Cache:", text)
        self.assertIn("built-in", text)
        self.assertIn("cache", text)
        self.assertIn("user", text)

    def test_doctor_shows_runtime_and_peak(self):
        selected = {
            "id": "qwen", "alias": "qwen", "context": 131072,
            "path": "/nonexistent/qwen.gguf", "sampling": {},
            "runtime": {
                "kv_type_k": "q8_0", "kv_type_v": "q8_0", "flash_attn": "on",
                "batch": 2048, "ubatch": 512, "parallel": 1,
                "cache_prompt": True, "cache_reuse": 256, "fit": True,
                "fit_target": 2048, "kv_unified": True, "no_kv_offload": False,
            },
        }
        with mock.patch.object(cli.usage, "peak_tokens", return_value=4096):
            text = self._doctor(selected=selected)
        self.assertIn("Runtime:", text)
        self.assertIn("q8_0 KV", text)
        self.assertIn("2048/512", text)
        self.assertIn("fit 2048", text)
        self.assertIn("cache-reuse 256", text)
        self.assertIn("peak 4K", text)

    def test_doctor_runtime_summary_includes_spec_and_cmoe(self):
        selected = {
            "id": "qwen", "alias": "qwen", "context": 131072,
            "path": "/nonexistent/qwen.gguf", "sampling": {},
            "runtime": {
                "kv_type_k": "q8_0", "kv_type_v": "q8_0", "flash_attn": "on",
                "batch": 2048, "ubatch": 512, "parallel": 1,
                "cache_prompt": True, "cache_reuse": 256, "fit": True,
                "fit_target": 2048, "kv_unified": True, "no_kv_offload": False,
                "spec_type": "draft-simple", "draft_model": "/tmp/draft.gguf",
                "cmoe": True, "reasoning": "off",
            },
        }
        with mock.patch.object(cli.usage, "peak_tokens", return_value=None):
            text = self._doctor(selected=selected)
        self.assertIn("spec draft-simple", text)
        self.assertIn("cmoe", text)
        self.assertIn("reasoning off", text)


class ContextCliTest(unittest.TestCase):
    GIB = 1024 ** 3

    def _machine(self, **kwargs):
        values = {
            "chip": "Apple M1", "model": "arm64",
            "total_ram": 16 * self.GIB, "unified": True, "gpu": "unknown",
            "vram": None, "disk_free": 100 * self.GIB,
            "available_ram": 8 * self.GIB, "gpus": [],
        }
        values.update(kwargs)
        return hardware.Hardware(**values)

    def _run(self, argv):
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_context_lists_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)):
                code, out, _ = self._run(["context"])
        self.assertEqual(code, 0)
        for alias in ("qwen2.5-7b-instruct-graphify", "glm-4.7-flash",
                      "inkling-2-mini"):
            self.assertIn(alias, out)
        self.assertIn("32K", out)
        self.assertIn("8K", out)

    def test_context_shows_model_value_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)):
                code, out, _ = self._run(["context", "qwen2.5-7b-instruct"])
        self.assertEqual(code, 0)
        self.assertIn("qwen2.5-7b-instruct-graphify", out)
        self.assertIn("8K", out)
        self.assertIn("built-in", out)

    def test_context_sets_builtin_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)):
                code, _, _ = self._run(
                    ["context", "qwen2.5-7b-instruct", "65536"])
            self.assertEqual(code, 0)
            data = json.loads(models.read_text())
        self.assertEqual(data["overrides"][config.DEFAULT]["context"], 65536)

    def test_context_auto_writes_choice(self):
        machine = self._machine(total_ram=128 * self.GIB,
                                available_ram=128 * self.GIB)
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)), \
                 mock.patch.object(cli, "_local_size",
                                   return_value=4 * self.GIB), \
                 mock.patch.object(cli.gguf, "read_metadata", return_value=None), \
                 mock.patch.object(cli.hardware, "detect_hardware",
                                   return_value=machine):
                code, _, _ = self._run(
                    ["context", "qwen2.5-7b-instruct", "auto"])
            self.assertEqual(code, 0)
            data = json.loads(models.read_text())
        self.assertEqual(data["overrides"][config.DEFAULT]["context"], 131072)

    def test_context_auto_dry_run_does_not_write(self):
        machine = self._machine(total_ram=128 * self.GIB,
                                available_ram=128 * self.GIB)
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)), \
                 mock.patch.object(cli, "_local_size",
                                   return_value=4 * self.GIB), \
                 mock.patch.object(cli.gguf, "read_metadata", return_value=None), \
                 mock.patch.object(cli.hardware, "detect_hardware",
                                   return_value=machine):
                code, _, _ = self._run(
                    ["context", "qwen2.5-7b-instruct", "auto", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(models.exists())

    def test_context_warns_when_explicit_does_not_fit(self):
        machine = self._machine(total_ram=8 * self.GIB,
                                available_ram=8 * self.GIB)
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)), \
                 mock.patch.object(cli, "_local_size",
                                   return_value=4 * self.GIB), \
                 mock.patch.object(cli.gguf, "read_metadata", return_value=None), \
                 mock.patch.object(cli.hardware, "detect_hardware",
                                   return_value=machine):
                code, _, err = self._run(
                    ["context", "qwen2.5-7b-instruct", "131072"])
            self.assertEqual(code, 0)
            data = json.loads(models.read_text())
        self.assertIn("warning", err.lower())
        self.assertEqual(data["overrides"][config.DEFAULT]["context"], 131072)

    def test_context_rejects_non_positive_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", dict(config.PROFILES)):
                code, _, err = self._run(
                    ["context", "qwen2.5-7b-instruct", "0"])
            self.assertEqual(code, 1)
            self.assertFalse(models.exists())
            self.assertIn("positive integer", err)

    def test_context_shows_runtime_summary(self):
        profiles = {
            "m": {
                "id": "m", "alias": "m", "path": "/x/m.gguf",
                "context": 131072, "sampling": {},
                "runtime": {
                    "kv_type_k": "q8_0", "kv_type_v": "q8_0",
                    "batch": 2048, "ubatch": 512, "fit": True,
                    "fit_target": 2048, "cache_reuse": 256,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.json"
            with mock.patch.object(config, "USER_MODELS", models), \
                 mock.patch.object(config, "PROFILES", profiles):
                code, out, _ = self._run(["context", "m"])
        self.assertEqual(code, 0)
        self.assertIn("q8_0 KV", out)
        self.assertIn("2048/512", out)
        self.assertIn("fit 2048", out)
        self.assertIn("cache-reuse 256", out)


class ApplyCliTest(unittest.TestCase):
    GIB = 1024 ** 3

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.models = Path(temporary.name) / "models.json"
        self._mocks = {}

    def _machine(self, total_ram=None, available_ram=None,
                 unified=True, vram=None):
        if total_ram is None:
            total_ram = 128 * self.GIB
        return hardware.Hardware(
            chip="Apple M1", model="arm64", total_ram=total_ram,
            unified=unified, gpu="unknown", vram=vram,
            disk_free=100 * self.GIB,
            available_ram=total_ram if available_ram is None else available_ram,
            gpus=[])

    def _apply(self, argv, profiles, size_map, machine, running=None,
               server_context=None, identity="", sync_openencode=None,
               sync_pi=None, sync_codex=None, selected_profile=None,
               start_result=None, peak=None, choose_runtime=None):
        out = io.StringIO()
        err = io.StringIO()

        def local_size(profile):
            return size_map.get(profile["id"])

        patches = {
            "running": mock.patch.object(service, "running", return_value=running),
            "server_context": mock.patch.object(service, "server_context",
                                                return_value=server_context),
            "identity": mock.patch.object(service, "identity", return_value=identity),
            "stop": mock.patch.object(service, "stop"),
            "start": mock.patch.object(service, "start", return_value=start_result),
            "wait_ready": mock.patch.object(service, "wait_ready"),
            "oc_sync": mock.patch.object(cli.opencode, "sync",
                                         return_value=sync_openencode),
            "pi_sync": mock.patch.object(cli.pi, "sync", return_value=sync_pi),
            "codex_sync": mock.patch.object(cli.codex, "sync",
                                            return_value=sync_codex),
            "selected_profile": mock.patch.object(
                config, "selected_profile",
                return_value=selected_profile or {"id": "sel", "alias": "sel"}),
            "peak_tokens": mock.patch.object(cli.usage, "peak_tokens",
                                             return_value=peak),
        }
        if choose_runtime is not None:
            patches["choose_runtime"] = mock.patch.object(
                cli.runtime, "choose_runtime", return_value=choose_runtime)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(config, "migrate_state",
                                                  return_value=False))
            stack.enter_context(mock.patch.object(config, "PROFILES", profiles))
            stack.enter_context(mock.patch.object(config, "USER_MODELS", self.models))
            stack.enter_context(mock.patch.object(cli, "_local_size",
                                                  side_effect=local_size))
            stack.enter_context(mock.patch.object(cli.gguf, "read_metadata",
                                                  return_value=None))
            stack.enter_context(mock.patch.object(cli.hardware, "detect_hardware",
                                                  return_value=machine))
            stack.enter_context(mock.patch.object(
                service, "locked", lambda: contextlib.nullcontext()))
            for name, patcher in patches.items():
                self._mocks[name] = stack.enter_context(patcher)
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_apply_dry_run_writes_nothing(self):
        machine = self._machine()
        code, out, _ = self._apply(
            ["apply", "--dry-run"], dict(config.PROFILES),
            {config.DEFAULT: 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        self.assertFalse(self.models.exists())
        self.assertIn("\u2192", out)

    def test_apply_updates_and_reloads(self):
        machine = self._machine()
        out = io.StringIO()
        with mock.patch.object(config, "USER_MODELS", self.models), \
             mock.patch.object(config, "PROFILES", dict(config.PROFILES)), \
             mock.patch.object(cli, "_local_size", return_value=4 * self.GIB), \
             mock.patch.object(cli.gguf, "read_metadata", return_value=None), \
             mock.patch.object(cli.hardware, "detect_hardware",
                               return_value=machine), \
             mock.patch.object(service, "running", return_value=None), \
             mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(service, "locked",
                               lambda: contextlib.nullcontext()), \
             redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["apply", "qwen2.5-7b-instruct"]), 0)
            self.assertEqual(config.PROFILES[config.DEFAULT]["context"], 131072)
        data = json.loads(self.models.read_text())
        self.assertEqual(data["overrides"][config.DEFAULT]["context"], 131072)

    def test_apply_context_is_ceiling(self):
        machine = self._machine()
        code, _, _ = self._apply(
            ["apply", "--context", "65536"], dict(config.PROFILES),
            {config.DEFAULT: 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        self.assertEqual(data["overrides"][config.DEFAULT]["context"], 65536)

    def test_apply_skips_missing_models(self):
        machine = self._machine()
        profiles = {
            "installed": {"id": "installed", "alias": "installed",
                          "path": "/x/installed.gguf", "context": 8192,
                          "sampling": {}, "repo": "o/r", "filename": "a.gguf",
                          "revision": "main"},
            "gone": {"id": "gone", "alias": "gone", "path": "/x/gone.gguf",
                     "context": 8192, "sampling": {}, "repo": "o/r2",
                     "filename": "g.gguf", "revision": "main"},
        }
        code, out, _ = self._apply(
            ["apply"], profiles, {"installed": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        self.assertIn("missing", out)
        data = json.loads(self.models.read_text())
        self.assertIn("installed", data["overrides"])
        self.assertNotIn("gone", data["overrides"])

    def test_apply_restarts_when_running_context_changed(self):
        machine = self._machine(total_ram=24 * self.GIB,
                                available_ram=24 * self.GIB)
        identity = ("llama-server -m /x.gguf "
                    "--alias qwen2.5-7b-instruct-graphify --port 18081")
        code, out, _ = self._apply(
            ["apply", "qwen2.5-7b-instruct"], dict(config.PROFILES),
            {config.DEFAULT: 4 * self.GIB}, machine,
            running=123, server_context=8192, identity=identity,
            start_result=(4242, True, {"alias": "qwen", "id": config.DEFAULT}))
        self.assertEqual(code, 0)
        self._mocks["stop"].assert_called_once()
        self._mocks["start"].assert_called_once()
        self._mocks["wait_ready"].assert_called_once()
        self.assertIn("restarted PID 4242", out)

    def test_apply_no_restart_prints_hint(self):
        machine = self._machine(total_ram=24 * self.GIB,
                                available_ram=24 * self.GIB)
        identity = ("llama-server -m /x.gguf "
                    "--alias qwen2.5-7b-instruct-graphify --port 18081")
        code, out, _ = self._apply(
            ["apply", "qwen2.5-7b-instruct", "--no-restart"],
            dict(config.PROFILES), {config.DEFAULT: 4 * self.GIB}, machine,
            running=123, server_context=8192, identity=identity)
        self.assertEqual(code, 0)
        self._mocks["stop"].assert_not_called()
        self._mocks["start"].assert_not_called()
        self._mocks["wait_ready"].assert_not_called()
        self.assertIn("llm restart", out)

    def test_apply_syncs_opencode_with_flag(self):
        machine = self._machine()
        profiles = {
            "m1": {"id": "m1", "alias": "m1", "path": "/x/m1.gguf",
                   "context": 8192, "sampling": {}, "repo": "o/r",
                   "filename": "m1.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--sync-opencode"], profiles, {"m1": 4 * self.GIB},
            machine,
            sync_openencode=cli.opencode.SyncResult(path=Path("/x/oc.json")),
            sync_pi=cli.pi.SyncResult(path=Path("/x/pi.json")))
        self.assertEqual(code, 0)
        self._mocks["oc_sync"].assert_called_once()
        self._mocks["pi_sync"].assert_not_called()
        self._mocks["codex_sync"].assert_not_called()
        installed = self._mocks["oc_sync"].call_args[0][0]
        self.assertIn("m1", installed)
        self.assertEqual(installed["m1"]["context"], 131072)

    def test_apply_syncs_pi_with_flag(self):
        machine = self._machine()
        profiles = {
            "m1": {"id": "m1", "alias": "m1", "path": "/x/m1.gguf",
                   "context": 8192, "sampling": {}, "repo": "o/r",
                   "filename": "m1.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--sync-pi"], profiles, {"m1": 4 * self.GIB}, machine,
            sync_openencode=cli.opencode.SyncResult(path=Path("/x/oc.json")),
            sync_pi=cli.pi.SyncResult(path=Path("/x/pi.json")))
        self.assertEqual(code, 0)
        self._mocks["oc_sync"].assert_not_called()
        self._mocks["pi_sync"].assert_called_once()
        self._mocks["codex_sync"].assert_not_called()

    def test_apply_syncs_both_with_sync_clients(self):
        machine = self._machine()
        profiles = {
            "m1": {"id": "m1", "alias": "m1", "path": "/x/m1.gguf",
                   "context": 8192, "sampling": {}, "repo": "o/r",
                   "filename": "m1.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--sync-clients"], profiles, {"m1": 4 * self.GIB},
            machine,
            sync_openencode=cli.opencode.SyncResult(path=Path("/x/oc.json")),
            sync_pi=cli.pi.SyncResult(path=Path("/x/pi.json")),
            sync_codex=cli.codex.SyncResult(path=Path("/x/codex.toml")))
        self.assertEqual(code, 0)
        self._mocks["oc_sync"].assert_called_once()
        self._mocks["pi_sync"].assert_called_once()
        self._mocks["codex_sync"].assert_called_once()
        selected_alias = self._mocks["codex_sync"].call_args[0][1]
        self.assertEqual(selected_alias, "sel")

    def test_apply_syncs_codex_with_flag(self):
        machine = self._machine()
        profiles = {
            "m1": {"id": "m1", "alias": "m1", "path": "/x/m1.gguf",
                   "context": 8192, "sampling": {}, "repo": "o/r",
                   "filename": "m1.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--sync-codex"], profiles, {"m1": 4 * self.GIB}, machine,
            sync_openencode=cli.opencode.SyncResult(path=Path("/x/oc.json")),
            sync_pi=cli.pi.SyncResult(path=Path("/x/pi.json")),
            sync_codex=cli.codex.SyncResult(path=Path("/x/codex.toml")),
            selected_profile={"id": "m1", "alias": "m1"})
        self.assertEqual(code, 0)
        self._mocks["oc_sync"].assert_not_called()
        self._mocks["pi_sync"].assert_not_called()
        self._mocks["codex_sync"].assert_called_once()
        self.assertEqual(self._mocks["codex_sync"].call_args[0][1], "m1")

    def test_apply_default_does_not_touch_clients(self):
        machine = self._machine()
        profiles = {
            "m1": {"id": "m1", "alias": "m1", "path": "/x/m1.gguf",
                   "context": 8192, "sampling": {}, "repo": "o/r",
                   "filename": "m1.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply"], profiles, {"m1": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        self._mocks["oc_sync"].assert_not_called()
        self._mocks["pi_sync"].assert_not_called()
        self._mocks["codex_sync"].assert_not_called()

    def test_apply_errors_on_named_missing_model(self):
        machine = self._machine()
        profiles = {
            "gone": {"id": "gone", "alias": "gone", "path": "/x/gone.gguf",
                     "context": 8192, "sampling": {}, "repo": "o/r",
                     "filename": "g.gguf", "revision": "main"},
        }
        code, _, err = self._apply(["apply", "gone"], profiles, {}, machine)
        self.assertEqual(code, 1)
        self.assertIn("not installed", err)

    def _choice(self, context=131072, runtime_dict=None, reason="fits",
                required=30000000000, budget=None, approximate=False,
                native=None):
        if runtime_dict is None:
            runtime_dict = {
                "kv_type_k": "q8_0", "kv_type_v": "q8_0", "flash_attn": "on",
                "batch": 2048, "ubatch": 512, "parallel": 1,
                "cache_prompt": True, "cache_reuse": 256, "fit": True,
                "fit_target": 2048, "kv_unified": True, "no_kv_offload": False,
            }
        return runtime.TunedChoice(context, runtime_dict, reason, required,
                                   budget, approximate, native)

    def test_apply_balanced_writes_runtime_and_context(self):
        machine = self._machine()
        choice = self._choice()
        code, out, _ = self._apply(
            ["apply", "qwen2.5-7b-instruct"], dict(config.PROFILES),
            {config.DEFAULT: 4 * self.GIB}, machine, choose_runtime=choice)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        override = data["overrides"][config.DEFAULT]
        self.assertEqual(override["context"], 131072)
        self.assertEqual(override["runtime"]["kv_type_k"], "q8_0")
        self.assertEqual(override["runtime"]["batch"], 2048)
        self.assertIn("q8_0", out)
        self.assertIn("2048/512", out)

    def test_apply_policy_conservative_and_aggressive(self):
        machine = self._machine()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--runtime-policy", "conservative"], profiles,
            {"m": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        self.assertEqual(data["overrides"]["m"]["runtime"]["kv_type_k"], "f16")

        code, _, _ = self._apply(
            ["apply", "--policy", "aggressive"], profiles,
            {"m": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        self.assertEqual(data["overrides"]["m"]["runtime"]["kv_type_k"], "q4_0")

    def test_apply_kv_override(self):
        machine = self._machine()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--kv", "q5_1"], profiles, {"m": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        self.assertEqual(data["overrides"]["m"]["runtime"]["kv_type_k"], "q5_1")
        self.assertEqual(data["overrides"]["m"]["runtime"]["kv_type_v"], "q5_1")

    def test_apply_uses_observed_peak(self):
        machine = self._machine()
        choice = self._choice()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply"], profiles, {"m": 4 * self.GIB}, machine,
            peak=40000, choose_runtime=choice)
        self.assertEqual(code, 0)
        kwargs = self._mocks["choose_runtime"].call_args.kwargs
        self.assertEqual(kwargs["observed"], 40000)
        self.assertEqual(kwargs["policy"], "balanced")

    def test_apply_dry_run_writes_no_runtime(self):
        machine = self._machine()
        choice = self._choice()
        with mock.patch.object(config, "set_tuning") as set_tuning:
            code, _, _ = self._apply(
                ["apply", "--dry-run"], dict(config.PROFILES),
                {config.DEFAULT: 4 * self.GIB}, machine, choose_runtime=choice)
        self.assertEqual(code, 0)
        set_tuning.assert_not_called()
        self.assertFalse(self.models.exists())

    def test_apply_restarts_on_runtime_change(self):
        machine = self._machine(total_ram=128 * self.GIB,
                                available_ram=128 * self.GIB)
        profiles = {
            config.DEFAULT: dict(config.PROFILES[config.DEFAULT], context=131072),
        }
        choice = self._choice(context=131072)
        identity = ("llama-server -m /x.gguf "
                    "--alias qwen2.5-7b-instruct-graphify --port 18081")
        code, out, _ = self._apply(
            ["apply", "qwen2.5-7b-instruct"], profiles,
            {config.DEFAULT: 4 * self.GIB}, machine,
            running=123, server_context=131072, identity=identity,
            start_result=(4242, True, {"alias": "qwen", "id": config.DEFAULT}),
            choose_runtime=choice)
        self.assertEqual(code, 0)
        self._mocks["stop"].assert_called_once()
        self._mocks["start"].assert_called_once()
        self._mocks["wait_ready"].assert_called_once()
        self.assertIn("restarted PID 4242", out)

    def test_apply_no_cache_reuse_and_no_fit(self):
        machine = self._machine()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--no-fit", "--no-cache-reuse"], profiles,
            {"m": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        tuned = data["overrides"]["m"]["runtime"]
        self.assertFalse(tuned["fit"])
        self.assertEqual(tuned["cache_reuse"], 0)

    def test_apply_draft_resolves_to_path(self):
        machine = self._machine()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
            "draft": {"id": "draft", "alias": "draft",
                      "path": "/x/draft.gguf", "context": 8192,
                      "sampling": {}, "repo": "o/r2",
                      "filename": "draft.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--draft", "draft"], profiles,
            {"m": 4 * self.GIB, "draft": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        runtime_block = data["overrides"]["m"]["runtime"]
        # A draft without --spec-type defaults to draft-simple and stores path.
        self.assertEqual(runtime_block["draft_model"], "/x/draft.gguf")
        self.assertEqual(runtime_block["spec_type"], "draft-simple")

    def test_apply_draft_explicit_spec_type_wins(self):
        machine = self._machine()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
            "draft": {"id": "draft", "alias": "draft",
                      "path": "/x/draft.gguf", "context": 8192,
                      "sampling": {}, "repo": "o/r2",
                      "filename": "draft.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--draft", "draft", "--spec-type", "draft-mtp"], profiles,
            {"m": 4 * self.GIB, "draft": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        runtime_block = data["overrides"]["m"]["runtime"]
        self.assertEqual(runtime_block["draft_model"], "/x/draft.gguf")
        self.assertEqual(runtime_block["spec_type"], "draft-mtp")

    def test_apply_cmoe_reasoning_device_flags(self):
        machine = self._machine()
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}, "repo": "o/r",
                  "filename": "m.gguf", "revision": "main"},
        }
        code, _, _ = self._apply(
            ["apply", "--cmoe", "--reasoning", "on", "--reasoning-budget",
             "1024", "--device", "0,1", "--top-k", "80", "--predict", "512"],
            profiles, {"m": 4 * self.GIB}, machine)
        self.assertEqual(code, 0)
        data = json.loads(self.models.read_text())
        runtime_block = data["overrides"]["m"]["runtime"]
        self.assertTrue(runtime_block["cmoe"])
        self.assertEqual(runtime_block["reasoning"], "on")
        self.assertEqual(runtime_block["reasoning_budget"], 1024)
        self.assertEqual(runtime_block["devices"], "0,1")
        self.assertEqual(runtime_block["top_k"], 80)
        self.assertEqual(runtime_block["n_predict"], 512)


class AnalyzeCliTest(unittest.TestCase):
    def test_analyze_cli_context_override(self):
        profiles = {
            "m": {"id": "m", "alias": "m", "path": "/x/m.gguf",
                  "context": 8192, "sampling": {}},
        }
        captured = {}

        def fake_rows(profiles, context=65536, machine=None):
            captured["context"] = context
            return [{"model": "m", "alias": "m", "size": None,
                     "context": context, "f16": None, "q8_0": None,
                     "q4_0": None, "saving": None, "fits": None,
                     "native": None}]

        out = io.StringIO()
        with mock.patch.object(config, "PROFILES", profiles), \
             mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(cli.hardware, "detect_hardware"), \
             mock.patch.object(cli.analysis, "rows", side_effect=fake_rows), \
             redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["analyze", "--context", "131072"]), 0)
        self.assertEqual(captured["context"], 131072)
        self.assertIn("128K", out.getvalue())

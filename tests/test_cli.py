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

from llm import cli, config, hardware, hub, service


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
            "llm 1.1.1",
            "last version is 1.1.1. you are up to date.",
        ])

    def test_version_uses_invocation_name(self):
        with mock.patch.object(sys, "argv", ["/Users/x/.local/bin/lmm"]):
            lines = self._output(["version"]).splitlines()
        self.assertEqual(lines[0], "lmm 1.1.1")

    def test_version_points_to_update_when_latest_is_newer(self):
        with mock.patch("llm.LATEST_VERSION", "1.2.0"):
            lines = self._output(["version"]).splitlines()
        self.assertEqual(lines, [
            "llm 1.1.1",
            "last version is 1.2.0. run `llm update` to update.",
        ])

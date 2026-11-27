import contextlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

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

    def test_get_registers_selected_file(self):
        files = [{"name": "M-Q4_K_M.gguf", "size": 10, "sha256": "a"}]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(hub, "model_files", return_value=files), \
             mock.patch.object(hub, "download_file",
                               return_value=Path(tmp) / "M-Q4_K_M.gguf"), \
             mock.patch.object(hub, "write_manifest"), \
             mock.patch.object(config, "register_model") as register, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["get", "owner/repo:Q4_K_M"]), 0)
        self.assertEqual(register.call_args[0][0]["path"],
                         str(Path(tmp) / "M-Q4_K_M.gguf"))

    def test_get_ambiguous_selector_lists_candidates(self):
        files = [{"name": "A-Q4.gguf", "size": 1, "sha256": "a"},
                 {"name": "A-Q5.gguf", "size": 2, "sha256": "b"}]
        buffer = io.StringIO()
        with mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(hub, "model_files", return_value=files), \
             redirect_stdout(io.StringIO()), redirect_stderr(buffer):
            self.assertEqual(cli.main(["get", "owner/repo"]), 1)
        self.assertIn("A-Q4.gguf", buffer.getvalue())
        self.assertIn("A-Q5.gguf", buffer.getvalue())

    def test_get_rejects_selector_plus_file(self):
        with mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(hub, "model_files", return_value=[]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["get", "owner/repo:Q4", "--file", "x.gguf"]), 1)

    def test_get_is_idempotent_for_registered_model(self):
        files = [{"name": "M-Q4.gguf", "size": 10, "sha256": "a"}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "models" / "existing"
            directory.mkdir(parents=True)
            path = directory / "M-Q4.gguf"
            path.write_bytes(b"x" * 10)
            existing = {"id": "existing", "alias": "existing", "repo": "owner/repo",
                        "path": str(path), "context": 8192, "sampling": {}}
            with mock.patch.object(config, "migrate_state", return_value=False), \
                 mock.patch.object(config, "PROFILES", {"existing": existing}), \
                 mock.patch.object(hub, "model_files", return_value=files), \
                 mock.patch.object(hub, "download_file", return_value=path) as download, \
                 mock.patch.object(hub, "write_manifest"), \
                 mock.patch.object(config, "register_model") as register, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["get", "owner/repo"]), 0)
            self.assertEqual(download.call_args[0][2], directory)
            self.assertEqual(register.call_args[0][0]["id"], "existing")

    def test_get_refuses_when_hardware_too_small(self):
        files = [{"name": "M-Q4.gguf", "size": 10, "sha256": "a"}]
        with mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(config, "PROFILES", {}), \
             mock.patch.object(hub, "model_files", return_value=files), \
             mock.patch.object(hardware, "preflight", return_value=["model does not fit"]), \
             mock.patch.object(hub, "download_file") as download, \
             mock.patch.object(hub, "write_manifest"), \
             mock.patch.object(config, "register_model"), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["get", "owner/repo:Q4"]), 1)
        download.assert_not_called()

    def test_get_force_downloads_when_hardware_refuses(self):
        files = [{"name": "M-Q4.gguf", "size": 10, "sha256": "a"}]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(config, "migrate_state", return_value=False), \
             mock.patch.object(config, "PROFILES", {}), \
             mock.patch.object(hub, "model_files", return_value=files), \
             mock.patch.object(hardware, "preflight", return_value=["model does not fit"]), \
             mock.patch.object(hub, "download_file",
                               return_value=Path(tmp) / "M-Q4.gguf") as download, \
             mock.patch.object(hub, "write_manifest"), \
             mock.patch.object(config, "register_model"), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["get", "owner/repo:Q4", "--force"]), 0)
        download.assert_called_once()

    def test_registered_model_matches_full_relative_filename(self):
        profiles = {"a": {"repo": "o/r", "path": "/m/a/M.gguf",
                          "filename": "a/M.gguf"}}
        self.assertIsNone(cli._registered_model(profiles, "o/r", "b/M.gguf"))

    def test_get_is_idempotent_for_nested_filename(self):
        files = [{"name": "sub/M.gguf", "size": 10, "sha256": "a"}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "models" / "existing"
            path = directory / "sub" / "M.gguf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"x" * 10)
            existing = {"id": "existing", "alias": "existing", "repo": "owner/repo",
                        "path": str(path), "directory": str(directory),
                        "filename": "sub/M.gguf", "context": 8192, "sampling": {}}
            with mock.patch.object(config, "migrate_state", return_value=False), \
                 mock.patch.object(config, "PROFILES", {"existing": existing}), \
                 mock.patch.object(hub, "model_files", return_value=files), \
                 mock.patch.object(hub, "download_file", return_value=path) as download, \
                 mock.patch.object(hub, "write_manifest"), \
                 mock.patch.object(config, "register_model") as register, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["get", "owner/repo"]), 0)
            self.assertEqual(download.call_args[0][2], directory)
            self.assertEqual(register.call_args[0][0]["id"], "existing")

    def test_get_existing_skips_preflight_without_force(self):
        files = [{"name": "M-Q4.gguf", "size": 10, "sha256": "a"}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "models" / "existing"
            directory.mkdir(parents=True)
            path = directory / "M-Q4.gguf"
            path.write_bytes(b"x" * 10)
            existing = {"id": "existing", "alias": "existing", "repo": "owner/repo",
                        "path": str(path), "directory": str(directory),
                        "filename": "M-Q4.gguf", "context": 8192, "sampling": {}}
            with mock.patch.object(config, "migrate_state", return_value=False), \
                 mock.patch.object(config, "PROFILES", {"existing": existing}), \
                 mock.patch.object(hub, "model_files", return_value=files), \
                 mock.patch.object(hardware, "preflight") as preflight, \
                 mock.patch.object(hub, "download_file", return_value=path), \
                 mock.patch.object(hub, "write_manifest"), \
                 mock.patch.object(config, "register_model"), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["get", "owner/repo"]), 0)
        preflight.assert_not_called()


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

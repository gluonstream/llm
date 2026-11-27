import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import service

class LogTest(unittest.TestCase):
    def test_tail_reads_only_the_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("old\n" + "x" * 1000 + "\nlast line\n")
            text = service.tail_text(path, max_bytes=64)
            self.assertTrue(text.endswith("last line\n"))
            self.assertLess(len(text), 100)

    def test_rotate_at_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("a" * 200)
            self.assertTrue(service.rotate_log(path, max_bytes=100))
            self.assertFalse(path.exists())
            self.assertEqual((Path(tmp) / "server.log.1").stat().st_size, 200)
            self.assertFalse(service.rotate_log(Path(tmp) / "server.log.1", max_bytes=100))

    def test_last_failure_uses_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("E llama_model_load: error loading model: "
                            "unknown model architecture: 'inkling2_loop'\n")
            self.assertIn("1.1.0", service.last_failure(path))

class ArgsTest(unittest.TestCase):
    def test_server_args_carry_profile_values(self):
        profile = {"id": "x", "alias": "x-alias", "path": "/tmp/x.gguf", "context": 4096,
                   "sampling": {"temperature": 0.7, "top_p": 1.0, "min_p": 0.01,
                                "repeat_penalty": 1.0}}
        args = service.server_args(profile)
        self.assertIn("/tmp/x.gguf", args)
        self.assertIn("4096", args)
        self.assertIn("x-alias", args)
        self.assertIn("--port", args)

    def test_server_args_uses_config_llama_server(self):
        profile = {"id": "x", "alias": "x", "path": "/tmp/x.gguf", "context": 1,
                   "sampling": {"temperature": 0, "top_p": 1, "min_p": 0,
                                "repeat_penalty": 1}}
        with mock.patch.object(service.config, "LLAMA_SERVER", "/custom/llama-server"):
            self.assertEqual(service.server_args(profile)[0], "/custom/llama-server")

    def test_identity_uses_config_ps(self):
        with mock.patch.object(service.config, "PS", "/custom/ps"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "identity"
            service.identity(123)
        self.assertEqual(run.call_args[0][0][0], "/custom/ps")

    def test_listener_uses_config_lsof(self):
        with mock.patch.object(service.config, "LSOF", "/custom/lsof"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "42\n"
            self.assertEqual(service.listener_pid(), 42)
        self.assertEqual(run.call_args[0][0][0], "/custom/lsof")

class SafetyTest(unittest.TestCase):
    def test_stale_pid_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "process.json"
            pidfile.write_text(json.dumps({"pid": os.getpid(), "identity": "old-identity"}))
            with mock.patch.object(service, "PIDFILE", pidfile), \
                 mock.patch.object(service, "identity", return_value="new-identity"):
                self.assertIsNone(service.running())

    def test_foreign_listener_is_not_expected(self):
        profile = {"id": "x", "alias": "x-alias", "path": "/tmp/x.gguf"}
        with mock.patch.object(service, "selected_profile", return_value=profile), \
             mock.patch.object(service, "identity", return_value="/usr/bin/other-server"):
            self.assertFalse(service.is_expected_listener(999))

    def test_empty_model_file_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "model.gguf"
            empty.touch()
            with self.assertRaises(service.ModelLoadError):
                service.validate_profile_files({"id": "x", "path": str(empty)})

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import opencode

QWEN = {"id": "qwen", "alias": "qwen2.5-7b-instruct-graphify", "context": 8192}


def write_config(tmp, models=None, extra_provider=True):
    doc = {
        "provider": {
            "llamacpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llama.cpp Local (llm)",
                "options": {"baseURL": "http://127.0.0.1:18081/v1"},
                "models": models if models is not None else {
                    "qwen2.5-7b-instruct-graphify": {
                        "name": "Qwen2.5 7B Instruct Graphify (local, 64k)",
                        "limit": {"context": 65536, "output": 8192},
                    },
                },
            },
        }
    }
    if extra_provider:
        doc["provider"]["deepseek"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "DeepSeek Direct",
            "options": {"baseURL": "https://api.deepseek.com"},
            "models": {"deepseek-flash": {"name": "DeepSeek Flash"}},
        }
    doc["model"] = "deepseek/deepseek-flash"
    path = Path(tmp) / "opencode.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def models_map(path):
    return json.loads(path.read_text())["provider"]["llamacpp"]["models"]


class OpenCodeConfigPathTest(unittest.TestCase):
    def test_config_path_uses_env(self):
        with mock.patch.dict(os.environ, {"LMM_OPENCODE_CONFIG": "/tmp/custom/oc.json"}):
            self.assertEqual(opencode.config_path(), Path("/tmp/custom/oc.json"))
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/xdg"}, clear=True):
            self.assertEqual(opencode.config_path(), Path("/xdg/opencode/opencode.json"))
        with mock.patch.dict(os.environ, {"HOME": "/home/me"}, clear=True):
            self.assertEqual(opencode.config_path(),
                             Path("/home/me/.config/opencode/opencode.json"))


class OpenCodeReadConfigTest(unittest.TestCase):
    def test_read_config_returns_none_for_missing_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opencode.json"
            self.assertIsNone(opencode.read_config(path))
            path.write_text("not json")
            self.assertIsNone(opencode.read_config(path))
            path.write_text("[1, 2, 3]")
            self.assertIsNone(opencode.read_config(path))

    def test_read_config_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            data = opencode.read_config(path)
            self.assertIsInstance(data, dict)
            self.assertIn("provider", data)


class OpenCodeSyncTest(unittest.TestCase):
    def test_sync_updates_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            result = opencode.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.updated, 1)
            self.assertEqual(result.added, 0)
            self.assertEqual(result.unchanged, 0)
            entry = models_map(path)["qwen2.5-7b-instruct-graphify"]
            self.assertEqual(entry["limit"]["context"], 8192)
            self.assertEqual(entry["limit"]["output"], 8192)
            self.assertEqual(entry["name"], "Qwen2.5 7B Instruct Graphify (local, 64k)")

    def test_sync_adds_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models={})
            result = opencode.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.added, 1)
            self.assertEqual(result.updated, 0)
            entry = models_map(path)["qwen2.5-7b-instruct-graphify"]
            self.assertEqual(entry, {"name": "qwen2.5-7b-instruct-graphify",
                                     "limit": {"context": 8192}})

    def test_sync_preserves_other_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            opencode.sync({"qwen": QWEN}, path=path)
            data = json.loads(path.read_text())
            self.assertIn("deepseek", data["provider"])
            self.assertIn("deepseek-flash", data["provider"]["deepseek"]["models"])
            self.assertEqual(data["model"], "deepseek/deepseek-flash")

    def test_sync_leaves_unknown_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models={
                "qwen2.5-7b-instruct-graphify": {
                    "name": "Qwen", "limit": {"context": 65536, "output": 8192}},
                "some-other-alias": {
                    "name": "Other", "limit": {"context": 4096}},
            })
            opencode.sync({"qwen": QWEN}, path=path)
            models = models_map(path)
            self.assertEqual(models["some-other-alias"]["limit"]["context"], 4096)
            self.assertEqual(models["qwen2.5-7b-instruct-graphify"]["limit"]["context"], 8192)

    def test_sync_writes_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            result = opencode.sync({"qwen": QWEN}, path=path)
            self.assertIsNotNone(result.backup)
            self.assertTrue(Path(result.backup).exists())
            backups = list(Path(tmp).glob("opencode.json.bak-*"))
            self.assertEqual(len(backups), 1)

    def test_sync_no_write_when_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models={
                "qwen2.5-7b-instruct-graphify": {
                    "name": "Qwen", "limit": {"context": 8192, "output": 8192}},
            })
            original = path.read_text()
            result = opencode.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.unchanged, 1)
            self.assertIsNone(result.backup)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(tmp).glob("opencode.json.bak-*")), [])

    def test_sync_skips_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opencode.json"
            path.write_text("not json")
            result = opencode.sync({"qwen": QWEN}, path=path)
            self.assertTrue(result.skipped)
            self.assertEqual(path.read_text(), "not json")
            self.assertEqual(list(Path(tmp).glob("*.bak-*")), [])

    def test_sync_skips_when_provider_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opencode.json"
            path.write_text(json.dumps({"model": "deepseek/deepseek-flash"}) + "\n")
            result = opencode.sync({"qwen": QWEN}, path=path)
            self.assertTrue(result.skipped)
            self.assertEqual(json.loads(path.read_text()),
                             {"model": "deepseek/deepseek-flash"})

    def test_sync_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            original = path.read_text()
            result = opencode.sync({"qwen": QWEN}, path=path, dry_run=True)
            self.assertEqual(result.updated, 1)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(tmp).glob("opencode.json.bak-*")), [])


class OpenCodeMismatchTest(unittest.TestCase):
    def test_mismatches_reports_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            self.assertEqual(
                opencode.mismatches({"qwen": QWEN}, path=path),
                [("qwen2.5-7b-instruct-graphify", 65536, 8192)])

    def test_mismatches_reports_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models={})
            self.assertEqual(
                opencode.mismatches({"qwen": QWEN}, path=path),
                [("qwen2.5-7b-instruct-graphify", None, 8192)])

    def test_mismatches_empty_when_in_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models={
                "qwen2.5-7b-instruct-graphify": {"limit": {"context": 8192}}})
            self.assertEqual(opencode.mismatches({"qwen": QWEN}, path=path), [])

    def test_unknown_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models={
                "qwen2.5-7b-instruct-graphify": {"limit": {"context": 8192}},
                "other-alias": {"limit": {"context": 4096}},
            })
            self.assertEqual(
                opencode.unknown_entries({"qwen": QWEN}, path=path),
                ["other-alias"])

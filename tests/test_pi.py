import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import config
from llm import pi

QWEN = {"id": "qwen", "alias": "qwen2.5-7b-instruct-graphify", "context": 65536}


def write_config(tmp, models=None, base_url=None, provider_name="llamacpp-local",
                 extra=True):
    providers = {
        provider_name: {
            "baseUrl": base_url if base_url is not None else config.URL + "/v1",
            "api": "openai-completions",
            "apiKey": "local",
            "models": models if models is not None else [
                {"id": "qwen2.5-7b-instruct-graphify", "name": "Local Qwen",
                 "contextWindow": 8192, "maxTokens": 2048},
            ],
        }
    }
    if extra:
        providers["openai"] = {
            "baseUrl": "https://api.openai.com/v1",
            "api": "openai-completions",
            "apiKey": "sk-...",
            "models": [{"id": "gpt-4o", "name": "GPT-4o",
                        "contextWindow": 128000, "maxTokens": 4096}],
        }
    doc = {"providers": providers}
    path = Path(tmp) / "models.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def provider_models(path):
    return json.loads(path.read_text())["providers"]["llamacpp-local"]["models"]


class PiConfigPathTest(unittest.TestCase):
    def test_config_path_uses_env(self):
        with mock.patch.dict(os.environ, {"PI_CODING_AGENT_DIR": "/tmp/pi-agent"}):
            self.assertEqual(pi.config_path(), Path("/tmp/pi-agent") / "models.json")
        with mock.patch.dict(os.environ, {"HOME": "/home/me"}, clear=True):
            self.assertEqual(pi.config_path(), Path("/home/me/.pi/agent/models.json"))


class PiReadConfigTest(unittest.TestCase):
    def test_read_config_returns_none_for_missing_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            self.assertIsNone(pi.read_config(path))
            path.write_text("not json")
            self.assertIsNone(pi.read_config(path))
            path.write_text("[1, 2, 3]")
            self.assertIsNone(pi.read_config(path))

    def test_read_config_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            self.assertIsInstance(pi.read_config(path), dict)


class PiSyncTest(unittest.TestCase):
    def test_sync_updates_context_window_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.updated, 1)
            self.assertEqual(result.added, 0)
            entry = provider_models(path)[0]
            self.assertEqual(entry["contextWindow"], 65536)
            self.assertEqual(entry["name"], "Local Qwen")
            self.assertEqual(entry["maxTokens"], 2048)

    def test_sync_adds_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models=[])
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.added, 1)
            self.assertEqual(result.updated, 0)
            entries = provider_models(path)
            self.assertEqual(entries, [{
                "id": "qwen2.5-7b-instruct-graphify",
                "name": "qwen2.5-7b-instruct-graphify",
                "contextWindow": 65536,
                "maxTokens": 2048,
            }])

    def test_sync_preserves_other_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            pi.sync({"qwen": QWEN}, path=path)
            data = json.loads(path.read_text())
            self.assertIn("openai", data["providers"])
            self.assertEqual(
                data["providers"]["openai"]["models"][0]["id"], "gpt-4o")

    def test_sync_requires_the_local_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, base_url="https://other.example/v1",
                                provider_name="other-provider", extra=False)
            original = path.read_text()
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertTrue(result.skipped)
            self.assertEqual(path.read_text(), original)

    def test_sync_finds_provider_by_base_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, provider_name="custom-provider")
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.updated, 1)
            entry = json.loads(path.read_text())["providers"]["custom-provider"]["models"][0]
            self.assertEqual(entry["contextWindow"], 65536)

    def test_sync_falls_back_to_llamacpp_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, base_url="https://other.example/v1",
                                provider_name="llamacpp-local", extra=False)
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.updated, 1)
            entry = provider_models(path)[0]
            self.assertEqual(entry["contextWindow"], 65536)

    def test_sync_writes_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertIsNotNone(result.backup)
            self.assertTrue(Path(result.backup).exists())
            backups = list(Path(tmp).glob("models.json.bak-*"))
            self.assertEqual(len(backups), 1)

    def test_sync_no_write_when_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models=[
                {"id": "qwen2.5-7b-instruct-graphify", "name": "Local Qwen",
                 "contextWindow": 65536, "maxTokens": 2048},
            ])
            original = path.read_text()
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertEqual(result.unchanged, 1)
            self.assertIsNone(result.backup)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(tmp).glob("models.json.bak-*")), [])

    def test_sync_skips_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text("not json")
            result = pi.sync({"qwen": QWEN}, path=path)
            self.assertTrue(result.skipped)
            self.assertEqual(path.read_text(), "not json")
            self.assertEqual(list(Path(tmp).glob("*.bak-*")), [])

    def test_sync_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            original = path.read_text()
            result = pi.sync({"qwen": QWEN}, path=path, dry_run=True)
            self.assertEqual(result.updated, 1)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(tmp).glob("models.json.bak-*")), [])


class PiMismatchTest(unittest.TestCase):
    def test_mismatches_reports_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            self.assertEqual(
                pi.mismatches({"qwen": QWEN}, path=path),
                [("qwen2.5-7b-instruct-graphify", 8192, 65536)])

    def test_mismatches_reports_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models=[])
            self.assertEqual(
                pi.mismatches({"qwen": QWEN}, path=path),
                [("qwen2.5-7b-instruct-graphify", None, 65536)])

    def test_mismatches_empty_when_in_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models=[
                {"id": "qwen2.5-7b-instruct-graphify", "name": "Local Qwen",
                 "contextWindow": 65536, "maxTokens": 2048},
            ])
            self.assertEqual(pi.mismatches({"qwen": QWEN}, path=path), [])

    def test_unknown_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, models=[
                {"id": "qwen2.5-7b-instruct-graphify", "name": "Local Qwen",
                 "contextWindow": 8192, "maxTokens": 2048},
                {"id": "other-model", "name": "Other",
                 "contextWindow": 8192, "maxTokens": 2048},
            ])
            self.assertEqual(pi.unknown_entries({"qwen": QWEN}, path=path),
                             ["other-model"])

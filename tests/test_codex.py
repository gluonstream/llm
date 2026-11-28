import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import codex, config

QWEN = {"id": "qwen", "alias": "qwen2.5-7b-instruct-graphify", "context": 8192}
ALIAS = "qwen2.5-7b-instruct-graphify"
BASE_URL = config.URL.rstrip("/") + "/v1"


def sample():
    return (
        "# codex config\n"
        "model = \"gpt-5-codex\"\n"
        "model_provider = \"openai\"\n"
        "\n"
        "[model_providers.openai]\n"
        "name = \"OpenAI\"\n"
        "base_url = \"https://api.openai.com/v1\"\n"
        "\n"
        "[marketplaces.my-mkt]\n"
        "source = \"https://example.com/mkt\"\n"
        "\n"
        "[plugins.my-plugin]\n"
        "source = \"https://example.com/plugin\"\n"
    )


def write_config(tmp, text=None):
    path = Path(tmp) / "config.toml"
    path.write_text(text if text is not None else sample())
    return path


class CodexConfigPathTest(unittest.TestCase):
    def test_config_path_uses_env(self):
        with mock.patch.dict(os.environ, {"CODEX_HOME": "/tmp/codex-home"}):
            self.assertEqual(codex.config_path(),
                             Path("/tmp/codex-home") / "config.toml")
        with mock.patch.dict(os.environ, {"HOME": "/home/me"}, clear=True):
            self.assertEqual(codex.config_path(),
                             Path("/home/me/.codex/config.toml"))


class CodexReadConfigTest(unittest.TestCase):
    def test_read_config_returns_none_for_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(codex.read_config(Path(tmp) / "config.toml"))

    def test_read_config_returns_raw_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            text = codex.read_config(path)
            self.assertIsInstance(text, str)
            self.assertIn("[marketplaces.my-mkt]", text)


class CodexSyncTest(unittest.TestCase):
    def test_sync_sets_model_and_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            codex.sync({"qwen": QWEN}, ALIAS, path=path)
            text = path.read_text()
            self.assertIn('model = "%s"' % ALIAS, text)
            self.assertIn('model_provider = "llamacpp"', text)

    def test_sync_writes_llamacpp_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            codex.sync({"qwen": QWEN}, ALIAS, path=path)
            text = path.read_text()
            self.assertIn("[model_providers.llamacpp]", text)
            self.assertIn('name = "llama.cpp"', text)
            self.assertIn('base_url = "%s"' % BASE_URL, text)
            self.assertIn('wire_api = "responses"', text)
            self.assertIn('requires_openai_auth = false', text)
            self.assertIn('supports_websockets = false', text)

    def test_sync_preserves_other_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            codex.sync({"qwen": QWEN}, ALIAS, path=path)
            text = path.read_text()
            self.assertIn("[marketplaces.my-mkt]", text)
            self.assertIn("[plugins.my-plugin]", text)
            self.assertIn('source = "https://example.com/mkt"', text)
            self.assertIn('source = "https://example.com/plugin"', text)
            self.assertIn("[model_providers.openai]", text)
            self.assertIn('name = "OpenAI"', text)

    def test_sync_updates_existing_llamacpp_section(self):
        text = (
            'model = "%s"\n' % ALIAS
            + 'model_provider = "llamacpp"\n'
            + "\n"
            + "[model_providers.llamacpp]\n"
            + 'name = "old name"\n'
            + 'base_url = "http://127.0.0.1:9931/v1"\n'
            + 'wire_api = "responses"\n'
            + 'requires_openai_auth = false\n'
            + 'supports_websockets = false\n'
            + 'custom_key = "keep me"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, text=text)
            result = codex.sync({"qwen": QWEN}, ALIAS, path=path)
            self.assertEqual(result.updated, 1)
            self.assertEqual(result.added, 0)
            out = path.read_text()
            self.assertIn('name = "llama.cpp"', out)
            self.assertIn('base_url = "%s"' % BASE_URL, out)
            self.assertIn('custom_key = "keep me"', out)

    def test_sync_is_idempotent_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            first = codex.sync({"qwen": QWEN}, ALIAS, path=path)
            self.assertIsNotNone(first.backup)
            written = path.read_text()
            again = codex.sync({"qwen": QWEN}, ALIAS, path=path)
            self.assertEqual(again.unchanged, 1)
            self.assertIsNone(again.backup)
            self.assertEqual(path.read_text(), written)
            self.assertEqual(list(Path(tmp).glob("config.toml.bak-*")),
                             [Path(first.backup)])

            fresh = Path(tmp) / "config2.toml"
            fresh.write_text(sample())
            before = fresh.read_text()
            dry = codex.sync({"qwen": QWEN}, ALIAS, path=fresh, dry_run=True)
            self.assertEqual(fresh.read_text(), before)
            self.assertIsNone(dry.backup)
            self.assertEqual(list(Path(tmp).glob("config2.toml.bak-*")), [])

    def test_sync_writes_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            result = codex.sync({"qwen": QWEN}, ALIAS, path=path)
            self.assertIsNotNone(result.backup)
            self.assertTrue(Path(result.backup).exists())
            self.assertEqual(len(list(Path(tmp).glob("config.toml.bak-*"))), 1)

    def test_sync_skips_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            result = codex.sync({"qwen": QWEN}, ALIAS, path=path)
            self.assertTrue(result.skipped)
            self.assertFalse(path.exists())


class CodexMismatchTest(unittest.TestCase):
    def test_mismatches_reports_model_and_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            mismatches = codex.mismatches(ALIAS, path=path)
            self.assertIn(("model", "gpt-5-codex", ALIAS), mismatches)
            self.assertIn(("model_provider", "openai", "llamacpp"), mismatches)
            self.assertIn(("base_url", None, BASE_URL), mismatches)

    def test_mismatches_empty_when_in_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            codex.sync({"qwen": QWEN}, ALIAS, path=path)
            self.assertEqual(codex.mismatches(ALIAS, path=path), [])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from llm import config

class ResolveModelTest(unittest.TestCase):
    def test_exact_alias_and_prefix_resolve(self):
        self.assertEqual(config.resolve_model("qwen2.5-7b-instruct")["id"],
                         "qwen2.5-7b-instruct-q4-k-m")
        self.assertEqual(config.resolve_model("qwen2.5-7b-instruct-q4-k-m")["id"],
                         "qwen2.5-7b-instruct-q4-k-m")
        self.assertEqual(config.resolve_model("inkling")["id"],
                         "inkling-2-mini-gguf-ud-q4-k-xl")

    def test_ambiguous_prefix_raises_with_candidates(self):
        profiles = {
            "glm-4.7-flash-ud-q4-k-xl": {"id": "glm-4.7-flash-ud-q4-k-xl", "alias": "glm-4.7-flash"},
            "glm-4.7-air-ud-q4-k-xl": {"id": "glm-4.7-air-ud-q4-k-xl", "alias": "glm-4.7-air"},
        }
        with self.assertRaises(config.ModelError) as caught:
            config.resolve_model("glm", profiles=profiles)
        self.assertEqual(sorted(caught.exception.candidates),
                         ["glm-4.7-air-ud-q4-k-xl", "glm-4.7-flash-ud-q4-k-xl"])

    def test_unknown_model_lists_choices(self):
        with self.assertRaises(config.ModelError) as caught:
            config.resolve_model("nope")
        self.assertIn("qwen2.5-7b-instruct-q4-k-m", caught.exception.candidates)

class SelectionTest(unittest.TestCase):
    def test_select_then_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selected-model.json"
            self.assertEqual(config.selected_profile(selection=selection)["id"], config.DEFAULT)
            config.select_profile("inkling", selection=selection)
            self.assertEqual(config.selected_profile(selection=selection)["id"],
                             "inkling-2-mini-gguf-ud-q4-k-xl")
            self.assertEqual(json.loads(selection.read_text())["model"],
                             "inkling-2-mini-gguf-ud-q4-k-xl")


class MigrationTest(unittest.TestCase):
    def test_migrate_moves_state_but_not_the_log(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy"
            state = Path(tmp) / "state"
            legacy.mkdir()
            (legacy / "selected-model.json").write_text('{"model": "x"}')
            (legacy / "label-cache-v3").mkdir()
            (legacy / "server.log").write_text("old log")
            with mock.patch.object(config, "STATE", state), \
                 mock.patch.object(config, "LEGACY_STATE", legacy):
                self.assertTrue(config.migrate_state())
                self.assertTrue((state / "selected-model.json").exists())
                self.assertTrue((state / "label-cache-v3").is_dir())
                self.assertFalse((state / "server.log").exists())
                self.assertTrue((legacy / "server.log").exists())
                self.assertFalse(config.migrate_state())


class RegistryTest(unittest.TestCase):
    def _profile(self, model_id="mine", alias="mine", path="/tmp/mine.gguf"):
        return {"id": model_id, "alias": alias, "path": path, "context": 8192,
                "sampling": {"temperature": 0.7, "top_p": 1.0,
                             "min_p": 0.01, "repeat_penalty": 1.0}}

    def test_load_merges_user_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({"mine": self._profile()}))
            merged = config.load_profiles(path=registry)
            self.assertIn(config.DEFAULT, merged)
            self.assertEqual(merged["mine"]["alias"], "mine")

    def test_register_writes_registry_and_updates_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            profiles = dict(config.BUILTIN_PROFILES)
            config.register_model(self._profile(), path=registry, profiles=profiles)
            self.assertEqual(json.loads(registry.read_text())["mine"]["alias"], "mine")
            self.assertIn("mine", profiles)

    def test_register_rejects_builtin_id_and_duplicate_alias(self):
        profiles = dict(config.BUILTIN_PROFILES)
        with self.assertRaises(config.ModelError):
            config.register_model(self._profile(model_id=config.DEFAULT, alias="x"),
                                  path=Path("/tmp/registry.json"), profiles=profiles)
        with self.assertRaises(config.ModelError):
            config.register_model(
                self._profile(model_id="other", alias="qwen2.5-7b-instruct-graphify"),
                path=Path("/tmp/registry.json"), profiles=profiles)

    def test_derive_names_are_unique(self):
        model_id, alias = config.derive_model_names(
            "unsloth/Inkling-2-Mini-GGUF", "Inkling-2-Mini-UD-Q4_K_XL.gguf",
            set(config.BUILTIN_PROFILES),
            {p["alias"] for p in config.BUILTIN_PROFILES.values()})
        self.assertNotIn(model_id, config.BUILTIN_PROFILES)
        self.assertNotIn(alias, {p["alias"] for p in config.BUILTIN_PROFILES.values()})

    def test_load_does_not_shadow_builtin_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({config.DEFAULT: {
                "id": config.DEFAULT, "alias": "shadow", "path": "/tmp/x",
                "context": 1, "sampling": {}}}))
            merged = config.load_profiles(path=registry)
            self.assertEqual(merged[config.DEFAULT]["alias"],
                             config.BUILTIN_PROFILES[config.DEFAULT]["alias"])

    def test_load_does_not_shadow_builtin_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            alias = config.BUILTIN_PROFILES[config.DEFAULT]["alias"]
            registry.write_text(json.dumps({"other": {
                "id": "other", "alias": alias, "path": "/tmp/x",
                "context": 1, "sampling": {}}}))
            merged = config.load_profiles(path=registry)
            self.assertNotIn("other", merged)

    def test_register_alias_is_case_insensitive(self):
        profiles = dict(config.BUILTIN_PROFILES)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(config.ModelError):
                config.register_model(
                    self._profile(model_id="other",
                                  alias="QWEN2.5-7B-INSTRUCT-GRAPHIFY"),
                    path=Path(tmp) / "models.json", profiles=profiles)

    def test_normalize_alias_slugs(self):
        self.assertEqual(config.normalize_alias("My Model!"), "my-model")

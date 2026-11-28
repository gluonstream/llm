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

COMMIT = "a" * 40


def cache_snapshot(root, repo, filename, commit=COMMIT):
    directory = root / ("models--" + repo.replace("/", "--"))
    (directory / "refs").mkdir(parents=True, exist_ok=True)
    (directory / "refs" / "main").write_text(commit)
    snapshot = directory / "snapshots" / commit / filename
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"x")
    return snapshot


def profile(model_id, alias, path="/tmp/model.gguf"):
    return {"id": model_id, "alias": alias, "path": path,
            "context": 8192, "sampling": {}}

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

    def _runtime(self, **overrides):
        runtime = {
            "kv_type_k": "q8_0",
            "kv_type_v": "q8_0",
            "flash_attn": "on",
            "batch": 2048,
            "ubatch": 512,
            "parallel": 1,
            "cache_prompt": True,
            "cache_reuse": 256,
            "fit": True,
            "fit_target": 2048,
            "kv_unified": True,
            "no_kv_offload": False,
        }
        runtime.update(overrides)
        return runtime

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

    def test_legacy_flat_registry_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({"mine": self._profile()}))
            loaded = config.load_registry(registry)
            self.assertIn("mine", loaded.models)
            self.assertEqual(loaded.models["mine"]["alias"], "mine")
            self.assertEqual(loaded.overrides, {})
            self.assertFalse(loaded.wrapped)

    def test_overrides_apply_to_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "overrides": {config.DEFAULT: {"context": 65536}}}))
            merged = config.load_profiles(path=registry, discovered={})
            self.assertEqual(merged[config.DEFAULT]["context"], 65536)

    def test_overrides_apply_to_user_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "models": {"mine": self._profile()},
                "overrides": {"mine": {"context": 131072}},
            }))
            merged = config.load_profiles(path=registry, discovered={})
            self.assertEqual(merged["mine"]["context"], 131072)

    def test_override_applies_to_discovered_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "overrides": {"cache-1": {"context": 65536}}}))
            merged = config.load_profiles(
                path=registry,
                discovered={"cache-1": self._profile(
                    "cache-1", "cache-one", path=str(Path(tmp) / "cache.gguf"))},
                warnings=[])
            self.assertEqual(merged["cache-1"]["context"], 65536)
            self.assertEqual(merged["cache-1"]["alias"], "cache-one")

    def test_invalid_override_is_ignored_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "overrides": {"x": {"context": "big"}}}))
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides, {})
            self.assertTrue(reg.invalid)

    def test_save_registry_wraps_when_overrides_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            reg = config.Registry({"mine": self._profile()},
                                  {config.DEFAULT: {"context": 65536}},
                                  False, [])
            config.save_registry(reg, registry)
            data = json.loads(registry.read_text())
            self.assertIn("models", data)
            self.assertIn("overrides", data)
            self.assertEqual(data["overrides"][config.DEFAULT]["context"], 65536)

    def test_register_model_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "models": {"mine": self._profile()},
                "overrides": {config.DEFAULT: {"context": 65536}},
            }))
            profiles = dict(config.BUILTIN_PROFILES)
            config.register_model(self._profile("second", "second"),
                                  path=registry, profiles=profiles)
            data = json.loads(registry.read_text())
            self.assertEqual(data["overrides"][config.DEFAULT]["context"], 65536)
            self.assertIn("second", data["models"])

    def test_set_context_writes_override_for_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            config.set_context(config.DEFAULT, 65536, path=registry)
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides[config.DEFAULT]["context"], 65536)

    def test_set_context_updates_user_model_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({"mine": self._profile()}))
            config.set_context("mine", 131072, path=registry)
            reg = config.load_registry(registry)
            self.assertEqual(reg.models["mine"]["context"], 131072)
            self.assertNotIn("mine", reg.overrides)

    def test_set_context_writes_override_for_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            discovered = self._profile(
                "cache-1", "cache-one", path=str(Path(tmp) / "cache.gguf"))
            with mock.patch.object(config, "PROFILES", {"cache-1": discovered}), \
                 mock.patch.object(config, "USER_MODELS", registry):
                config.set_context("cache-1", 131072)
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides["cache-1"]["context"], 131072)
            self.assertNotIn("cache-1", reg.models)

    def test_reload_profiles_refreshes_global(self):
        original_profiles = config.PROFILES
        original_warnings = config.PROFILE_WARNINGS
        try:
            with tempfile.TemporaryDirectory() as tmp:
                registry = Path(tmp) / "models.json"
                with mock.patch.object(config, "USER_MODELS", registry):
                    config.set_context(config.DEFAULT, 65536)
                    config.reload_profiles()
                    self.assertEqual(config.PROFILES[config.DEFAULT]["context"],
                                     65536)
        finally:
            config.PROFILES = original_profiles
            config.PROFILE_WARNINGS = original_warnings

    def test_context_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            self.assertEqual(config.context_source(config.DEFAULT, path=registry),
                             "built-in")
            registry.write_text(json.dumps({
                "overrides": {config.DEFAULT: {"context": 65536}}}))
            self.assertEqual(config.context_source(config.DEFAULT, path=registry),
                             "override")
            registry.write_text(json.dumps({"mine": self._profile()}))
            self.assertEqual(config.context_source("mine", path=registry), "user")
            self.assertEqual(config.context_source("nope", path=registry), "unknown")

    def test_runtime_override_valid_and_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            runtime = self._runtime()
            registry.write_text(json.dumps({
                "overrides": {config.DEFAULT: {
                    "context": 65536, "runtime": runtime}}}))
            merged = config.load_profiles(path=registry, discovered={})
            self.assertEqual(merged[config.DEFAULT]["context"], 65536)
            self.assertEqual(merged[config.DEFAULT]["runtime"], runtime)
            # The override's runtime is copied, not shared.
            runtime["batch"] = 1
            self.assertEqual(merged[config.DEFAULT]["runtime"]["batch"], 2048)

    def test_invalid_runtime_override_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "overrides": {"x": {
                    "context": 65536,
                    "runtime": {"kv_type_k": "q3_k", "flash_attn": "loud",
                                "batch": 0, "cache_reuse": 64}}}}))
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides, {})
            self.assertTrue(reg.invalid)

    def test_set_tuning_builtin_and_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            runtime = self._runtime()
            # Built-in id: written as an override.
            config.set_tuning(config.DEFAULT, 65536, runtime, path=registry)
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides[config.DEFAULT]["context"], 65536)
            self.assertEqual(reg.overrides[config.DEFAULT]["runtime"], runtime)
            # User id: written into the user model entry.
            registry.write_text(json.dumps({"mine": self._profile()}))
            config.set_tuning("mine", 131072, runtime, path=registry)
            reg = config.load_registry(registry)
            self.assertEqual(reg.models["mine"]["context"], 131072)
            self.assertEqual(reg.models["mine"]["runtime"], runtime)
            self.assertNotIn("mine", reg.overrides)

    def test_set_tuning_preserves_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            runtime = self._runtime()
            # A built-in that already has a context override keeps it when a
            # runtime is added.
            config.set_context(config.DEFAULT, 65536, path=registry)
            config.set_tuning(config.DEFAULT, 65536, runtime, path=registry)
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides[config.DEFAULT]["context"], 65536)
            self.assertEqual(reg.overrides[config.DEFAULT]["runtime"], runtime)

    def test_set_tuning_rejects_invalid_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            with self.assertRaises(config.ModelError):
                config.set_tuning(config.DEFAULT, 65536,
                                  {"kv_type_k": "q3_k"}, path=registry)

    def test_runtime_validation_new_fields(self):
        clean, errors = config._validate_runtime({
            "spec_type": "draft-mtp", "draft_model": "/tmp/draft.gguf",
            "cmoe": True, "reasoning": "on", "reasoning_budget": 1024,
            "devices": "0,1", "top_k": 80, "n_predict": 512,
        })
        self.assertEqual(errors, [])
        self.assertEqual(clean["spec_type"], "draft-mtp")
        self.assertEqual(clean["draft_model"], "/tmp/draft.gguf")
        self.assertTrue(clean["cmoe"])
        self.assertEqual(clean["reasoning"], "on")
        self.assertEqual(clean["reasoning_budget"], 1024)
        self.assertEqual(clean["devices"], "0,1")
        self.assertEqual(clean["top_k"], 80)
        self.assertEqual(clean["n_predict"], 512)

        # A list of devices is accepted; bad values are reported and dropped.
        clean, errors = config._validate_runtime({
            "devices": ["0", "1"], "spec_type": "bogus",
            "reasoning": "loud", "top_k": "many", "n_predict": 1.5,
            "reasoning_budget": -5, "draft_model": 3,
        })
        self.assertEqual(clean["devices"], "0,1")
        self.assertNotIn("spec_type", clean)
        self.assertNotIn("reasoning", clean)
        self.assertNotIn("top_k", clean)
        self.assertNotIn("n_predict", clean)
        self.assertNotIn("reasoning_budget", clean)
        self.assertNotIn("draft_model", clean)
        self.assertTrue(any("spec_type" in e for e in errors))
        self.assertTrue(any("reasoning" in e for e in errors))

    def test_set_tuning_persists_generation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            runtime = self._runtime(spec_type="draft-simple",
                                    draft_model="/tmp/draft.gguf",
                                    cmoe=True, reasoning="on",
                                    reasoning_budget=2048, devices="0,1",
                                    top_k=64, n_predict=256)
            config.set_tuning(config.DEFAULT, 65536, runtime, path=registry)
            reg = config.load_registry(registry)
            saved = reg.overrides[config.DEFAULT]["runtime"]
            self.assertEqual(saved["spec_type"], "draft-simple")
            self.assertEqual(saved["draft_model"], "/tmp/draft.gguf")
            self.assertTrue(saved["cmoe"])
            self.assertEqual(saved["reasoning"], "on")
            self.assertEqual(saved["reasoning_budget"], 2048)
            self.assertEqual(saved["devices"], "0,1")
            self.assertEqual(saved["top_k"], 64)
            self.assertEqual(saved["n_predict"], 256)

    def test_cache_reuse_zero_is_valid_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            config.set_tuning(config.DEFAULT, 65536,
                              self._runtime(cache_reuse=0), path=registry)
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides[config.DEFAULT]["runtime"]["cache_reuse"], 0)
            merged = config.load_profiles(path=registry, discovered={})
            self.assertEqual(merged[config.DEFAULT]["runtime"]["cache_reuse"], 0)

    def test_negative_cache_reuse_reported_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            registry.write_text(json.dumps({
                "overrides": {"x": {
                    "context": 65536,
                    "runtime": {"cache_reuse": -1}}}}))
            reg = config.load_registry(registry)
            self.assertEqual(reg.overrides, {})
            self.assertTrue(reg.invalid)

    def test_context_source_reports_cache_for_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            discovered = dict(self._profile(
                "cache-1", "cache-one", path=str(Path(tmp) / "cache.gguf")))
            discovered["source"] = "cache"
            with mock.patch.object(config, "PROFILES",
                                   {"cache-1": discovered}), \
                 mock.patch.object(config, "USER_MODELS",
                                   Path(tmp) / "models.json"):
                self.assertEqual(config.context_source("cache-1"), "cache")


class CacheProfileTest(unittest.TestCase):
    def _empty_registry(self, tmp):
        return Path(tmp) / "models.json"

    def test_builtin_profiles_resolve_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = cache_snapshot(
                root, "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
            with mock.patch.object(config, "HF_CACHE", root):
                merged = config.load_profiles(
                    path=self._empty_registry(tmp), discovered={})
            self.assertEqual(merged[config.DEFAULT]["path"], str(snapshot))

    def test_missing_builtin_keeps_expected_snapshot_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = config.BUILTIN_PROFILES[config.DEFAULT]
            expected = (root / "models--Qwen--Qwen2.5-7B-Instruct-GGUF"
                        / "snapshots" / "main" / builtin["filename"])
            with mock.patch.object(config, "HF_CACHE", root):
                merged = config.load_profiles(
                    path=self._empty_registry(tmp), discovered={})
            self.assertEqual(merged[config.DEFAULT]["path"], str(expected))

    def test_load_profiles_merges_discovered_before_user_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._empty_registry(tmp)
            discovered = {
                "qwen2.5-coder-keep": profile(
                    "qwen2.5-coder-keep", "qwen2.5-coder-keep",
                    path="/tmp/keep.gguf"),
                "qwen2.5-coder-lose": profile(
                    "qwen2.5-coder-lose", "shared-alias",
                    path="/tmp/lose.gguf"),
            }
            registry.write_text(json.dumps({
                "mine": profile("mine", "shared-alias", path="/tmp/mine.gguf"),
            }))
            merged = config.load_profiles(
                path=registry, discovered=discovered)
            self.assertIn("qwen2.5-coder-keep", merged)
            self.assertNotIn("qwen2.5-coder-lose", merged)
            self.assertIn("mine", merged)
            self.assertEqual(merged["mine"]["alias"], "shared-alias")

    def test_user_id_replaces_discovered_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._empty_registry(tmp)
            discovered = {"mine": profile("mine", "from-cache",
                                          path="/tmp/cache.gguf")}
            registry.write_text(json.dumps({
                "mine": profile("mine", "from-user", path="/tmp/user.gguf"),
            }))
            merged = config.load_profiles(
                path=registry, discovered=discovered)
            self.assertEqual(merged["mine"]["path"], "/tmp/user.gguf")

    def test_discovered_does_not_shadow_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._empty_registry(tmp)
            alias = config.BUILTIN_PROFILES[config.DEFAULT]["alias"]
            discovered = {
                "by-alias": profile("by-alias", alias),
                config.DEFAULT: profile(config.DEFAULT, "other-alias"),
            }
            merged = config.load_profiles(
                path=registry, discovered=discovered)
            self.assertNotIn("by-alias", merged)
            self.assertEqual(merged[config.DEFAULT]["alias"], alias)

    def test_discovery_does_not_duplicate_a_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = config.BUILTIN_PROFILES[config.DEFAULT]
            snapshot = cache_snapshot(
                root, builtin["repo"], builtin["filename"])
            registry = self._empty_registry(tmp)
            discovered = {
                "qwen2-5-7b-instruct-q4-k-m": {
                    "id": "qwen2-5-7b-instruct-q4-k-m",
                    "alias": "qwen2-5-7b-instruct-q4-k-m",
                    "repo": builtin["repo"],
                    "filename": builtin["filename"],
                    "revision": "main",
                    "path": str(snapshot),
                    "context": 8192,
                    "sampling": {},
                },
            }
            with mock.patch.object(config, "HF_CACHE", root):
                merged = config.load_profiles(
                    path=registry, discovered=discovered)
            self.assertNotIn("qwen2-5-7b-instruct-q4-k-m", merged)
            self.assertEqual(set(merged), set(config.BUILTIN_PROFILES))

    def test_discovery_skips_same_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = self._empty_registry(tmp)
            target = root / "shared.gguf"
            target.write_bytes(b"x")
            link = root / "linked.gguf"
            link.symlink_to(target)
            registry.write_text(json.dumps({
                "mine": profile("mine", "mine", path=str(target)),
            }))
            discovered = {
                "different-metadata": {
                    "id": "different-metadata",
                    "alias": "different-metadata",
                    "repo": "org/different",
                    "filename": "different.gguf",
                    "revision": "main",
                    "path": str(link),
                    "context": 8192,
                    "sampling": {},
                },
            }
            with mock.patch.object(config, "HF_CACHE", root):
                merged = config.load_profiles(
                    path=registry, discovered=discovered)
            self.assertNotIn("different-metadata", merged)
            self.assertIn("mine", merged)

    def test_discovery_keeps_distinct_cache_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = config.BUILTIN_PROFILES[config.DEFAULT]
            snapshot = cache_snapshot(
                root, builtin["repo"], builtin["filename"])
            registry = self._empty_registry(tmp)
            discovered = {
                "builtin-copy": {
                    "id": "builtin-copy",
                    "alias": "builtin-copy",
                    "repo": builtin["repo"],
                    "filename": builtin["filename"],
                    "revision": "main",
                    "path": str(snapshot),
                    "context": 8192,
                    "sampling": {},
                },
                "qwen2-5-coder-14b-instruct-q4-k-m": profile(
                    "qwen2-5-coder-14b-instruct-q4-k-m",
                    "qwen2-5-coder-14b-instruct-q4-k-m",
                    path=str(root / "coder.gguf")),
                "qwen3-8-flash-next-gsq-rco-q2-0": profile(
                    "qwen3-8-flash-next-gsq-rco-q2-0",
                    "qwen3-8-flash-next-gsq-rco-q2-0",
                    path=str(root / "flash.gguf")),
            }
            with mock.patch.object(config, "HF_CACHE", root):
                merged = config.load_profiles(
                    path=registry, discovered=discovered)
            self.assertNotIn("builtin-copy", merged)
            self.assertIn("qwen2-5-coder-14b-instruct-q4-k-m", merged)
            self.assertIn("qwen3-8-flash-next-gsq-rco-q2-0", merged)

    def test_lmm_no_cache_disables_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_snapshot(root, "org/coder", "coder-Q4_K_M.gguf")
            registry = self._empty_registry(tmp)
            with mock.patch.object(config, "HF_CACHE", root):
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("LMM_NO_CACHE", None)
                    enabled = config.load_profiles(path=registry)
                with mock.patch.dict(os.environ, {"LMM_NO_CACHE": "1"}):
                    disabled = config.load_profiles(path=registry)
            self.assertTrue(any(item.get("source") == "cache"
                                for item in enabled.values()))
            self.assertFalse(any(item.get("source") == "cache"
                                 for item in disabled.values()))


class WarningTest(unittest.TestCase):
    def test_dropped_user_alias_produces_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            alias = config.BUILTIN_PROFILES[config.DEFAULT]["alias"]
            registry.write_text(json.dumps({"other": {
                "id": "other", "alias": alias, "path": "/tmp/x",
                "context": 1, "sampling": {}}}))
            warnings = []
            merged = config.load_profiles(path=registry, discovered={},
                                          warnings=warnings)
            self.assertNotIn("other", merged)
            self.assertTrue(any("other" in w and "alias" in w
                                for w in warnings))

    def test_skipped_discovered_model_produces_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            alias = config.BUILTIN_PROFILES[config.DEFAULT]["alias"]
            discovered = {"dup": profile("dup", alias, path="/tmp/dup.gguf")}
            warnings = []
            config.load_profiles(path=registry, discovered=discovered,
                                 warnings=warnings)
            self.assertTrue(any("dup" in w and "alias" in w for w in warnings))

    def test_expected_dedupe_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = config.BUILTIN_PROFILES[config.DEFAULT]
            snapshot = cache_snapshot(root, builtin["repo"], builtin["filename"])
            registry = Path(tmp) / "models.json"
            discovered = {
                "qwen2-5-7b-instruct-q4-k-m": {
                    "id": "qwen2-5-7b-instruct-q4-k-m",
                    "alias": "qwen2-5-7b-instruct-q4-k-m",
                    "repo": builtin["repo"],
                    "filename": builtin["filename"],
                    "revision": "main",
                    "path": str(snapshot),
                    "context": 8192,
                    "sampling": {},
                },
            }
            warnings = []
            with mock.patch.object(config, "HF_CACHE", root):
                merged = config.load_profiles(
                    path=registry, discovered=discovered, warnings=warnings)
            self.assertNotIn("qwen2-5-7b-instruct-q4-k-m", merged)
            self.assertEqual(warnings, [])

    def test_dropped_user_model_still_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            alias = config.BUILTIN_PROFILES[config.DEFAULT]["alias"]
            registry.write_text(json.dumps({"other": {
                "id": "other", "alias": alias, "path": "/tmp/x",
                "context": 1, "sampling": {}}}))
            warnings = []
            merged = config.load_profiles(path=registry, discovered={},
                                          warnings=warnings)
            self.assertNotIn("other", merged)
            self.assertEqual(len(warnings), 1)
            self.assertIn("other", warnings[0])

    def test_dropped_user_profile_does_not_suppress_discovered_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            alias = config.BUILTIN_PROFILES[config.DEFAULT]["alias"]
            registry.write_text(json.dumps({"dropped": {
                "id": "dropped", "alias": alias,
                "path": "/tmp/shared.gguf", "context": 1, "sampling": {}}}))
            discovered = {"kept": profile("kept", "kept",
                                          path="/tmp/shared.gguf")}
            merged = config.load_profiles(path=registry,
                                          discovered=discovered)
            self.assertNotIn("dropped", merged)
            self.assertIn("kept", merged)

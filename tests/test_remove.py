import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import config, hfcache, remove, service

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40

DEFAULT_SAMPLING = {
    "temperature": 0.7, "top_p": 1.0, "min_p": 0.01, "repeat_penalty": 1.0,
}


def _blob(repo_dir, data):
    blob = repo_dir / "blobs" / hashlib.sha256(data).hexdigest()
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    return blob


def _link(repo, filename, commit, root, data):
    repo_dir = hfcache.repo_dir(repo, root)
    blob = _blob(repo_dir, data)
    link = hfcache.link_snapshot(repo, filename, commit, blob, root)
    return link, blob


def _split_links(repo, commit, root, stem="M-Q4", subdir="q4",
                 shard_data=(b"a", b"bb")):
    repo_dir = hfcache.repo_dir(repo, root)
    links, blobs = [], []
    total = len(shard_data)
    for index, data in enumerate(shard_data, start=1):
        filename = "%s/%s-%05d-of-%05d.gguf" % (subdir, stem, index, total)
        blob = _blob(repo_dir, data)
        link = hfcache.link_snapshot(repo, filename, commit, blob, root)
        links.append(link)
        blobs.append(blob)
    return links, blobs


def _split_two_snapshots(repo, root, stem="M-Q2", subdir="Q2_0",
                         shard_data=(b"a", b"bb")):
    """Create the same split set under two snapshot commits, sharing blobs."""
    repo_dir = hfcache.repo_dir(repo, root)
    links_by_commit = {COMMIT: [], OTHER_COMMIT: []}
    blobs = []
    total = len(shard_data)
    for index, data in enumerate(shard_data, start=1):
        filename = "%s/%s-%05d-of-%05d.gguf" % (subdir, stem, index, total)
        blob = _blob(repo_dir, data)
        for commit in (COMMIT, OTHER_COMMIT):
            link = hfcache.link_snapshot(repo, filename, commit, blob, root)
            links_by_commit[commit].append(link)
        blobs.append(blob)
    return links_by_commit, blobs


def _profile(model_id, alias, repo, filename, revision, path, source=None):
    profile = {
        "id": model_id, "alias": alias, "repo": repo, "filename": filename,
        "revision": revision, "path": str(path), "context": 8192,
        "sampling": dict(DEFAULT_SAMPLING),
    }
    if source:
        profile["source"] = source
    return profile


class RemoveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.models = self.root / "models.json"
        self._profiles = config.PROFILES
        self._warnings = config.PROFILE_WARNINGS
        self._hf = mock.patch.object(config, "HF_CACHE", self.root)
        self._hf.start()
        self._user = mock.patch.object(config, "USER_MODELS", self.models)
        self._user.start()
        self.addCleanup(self._restore)

    def _restore(self):
        self._user.stop()
        self._hf.stop()
        config.PROFILES = self._profiles
        config.PROFILE_WARNINGS = self._warnings

    def test_exact_name_only(self):
        profiles = {
            "m-alpha": {"id": "m-alpha", "alias": "alpha"},
            "m-beta": {"id": "m-beta", "alias": "beta"},
        }
        self.assertEqual(remove.resolve("alpha", profiles)["id"], "m-alpha")
        self.assertEqual(remove.resolve("M-ALPHA", profiles)["id"], "m-alpha")
        with self.assertRaises(config.ModelError) as caught:
            remove.resolve("al", profiles)
        self.assertIn("alpha", caught.exception.candidates)
        self.assertIn("beta", caught.exception.candidates)
        self.assertIn("exact", str(caught.exception))

    def test_plan_identifies_split_set_and_shared_blobs(self):
        repo = "org/model"
        links, blobs = _split_links(repo, COMMIT, self.root)
        _link(repo, "q8/M-Q8.gguf", COMMIT, self.root, b"cccc")
        profile = _profile("m-q4", "q4", repo, "q4/M-Q4-00001-of-00002.gguf",
                           "main", links[0])
        result = remove.plan(profile, root=self.root)
        self.assertEqual(result["repo"], repo)
        self.assertEqual(result["commit"], COMMIT)
        self.assertEqual(len(result["files"]), 2)
        filenames = sorted(item["filename"] for item in result["files"])
        self.assertEqual(filenames, [
            "q4/M-Q4-00001-of-00002.gguf",
            "q4/M-Q4-00002-of-00002.gguf",
        ])
        self.assertTrue(all(not item["shared"] for item in result["files"]))
        self.assertEqual(result["freed"], sum(len(data) for data in (b"a", b"bb")))

    def test_dry_run_is_default(self):
        repo = "org/model"
        links, blobs = _split_links(repo, COMMIT, self.root)
        profile = _profile("m-q4", "q4", repo, "q4/M-Q4-00001-of-00002.gguf",
                           "main", links[0], source="cache")
        result = remove.remove(profile, root=self.root)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["deleted"], [])
        for link in links:
            self.assertTrue(link.is_symlink())
        for blob in blobs:
            self.assertTrue(blob.exists())

    def test_apply_deletes_only_the_named_quant(self):
        repo = "org/model"
        repo_dir = hfcache.repo_dir(repo, self.root)
        links, blobs = _split_links(repo, COMMIT, self.root)
        q8_link, q8_blob = _link(repo, "q8/M-Q8.gguf", COMMIT, self.root, b"cccc")
        profile = _profile("m-q4", "q4", repo, "q4/M-Q4-00001-of-00002.gguf",
                           "main", links[0], source="cache")
        result = remove.remove(profile, root=self.root, apply=True)
        self.assertFalse(result["dry_run"])
        for link in links:
            self.assertFalse(link.exists())
        for blob in blobs:
            self.assertFalse(blob.exists())
        self.assertTrue(q8_link.exists())
        self.assertTrue(q8_blob.exists())
        self.assertTrue(repo_dir.exists())

    def test_apply_keeps_blob_shared_by_another_quant(self):
        repo = "org/model"
        data = b"shared-bytes"
        repo_dir = hfcache.repo_dir(repo, self.root)
        blob = _blob(repo_dir, data)
        link_q4 = hfcache.link_snapshot(repo, "q4/x.gguf", COMMIT, blob,
                                        self.root)
        link_q8 = hfcache.link_snapshot(repo, "q8/x.gguf", COMMIT, blob,
                                        self.root)
        profile = _profile("m-q4", "q4", repo, "q4/x.gguf", "main", link_q4,
                           source="cache")
        result = remove.remove(profile, root=self.root, apply=True)
        self.assertFalse(link_q4.exists())
        self.assertTrue(link_q8.exists())
        self.assertTrue(blob.exists())
        self.assertIn(str(blob), result["kept"])
        self.assertEqual(result["freed"], 0)

    def test_two_snapshots_same_model_removes_all_copies(self):
        repo = "org/model"
        links_by_commit, blobs = _split_two_snapshots(repo, self.root)
        profile = _profile("m-q2", "q2", repo, "Q2_0/M-Q2-00001-of-00002.gguf",
                           "main", links_by_commit[COMMIT][0])
        plan = remove.plan(profile, root=self.root)
        self.assertEqual(len(plan["files"]), 4)
        self.assertEqual({item["commit"] for item in plan["files"]},
                         {COMMIT, OTHER_COMMIT})
        self.assertTrue(all(not item["shared"] for item in plan["files"]))
        self.assertEqual(plan["freed"],
                         sum(len(data) for data in (b"a", b"bb")))

        result = remove.remove(profile, root=self.root, apply=True)
        self.assertEqual(result["freed"],
                         sum(len(data) for data in (b"a", b"bb")))
        for links in links_by_commit.values():
            for link in links:
                self.assertFalse(link.exists())
        for blob in blobs:
            self.assertFalse(blob.exists())

    def test_single_snapshot_model_frees_its_blob(self):
        repo = "org/model"
        link, blob = _link(repo, "m.gguf", COMMIT, self.root, b"gguf")
        profile = _profile("m", "m", repo, "m.gguf", "main", link,
                           source="cache")
        plan = remove.plan(profile, root=self.root)
        self.assertEqual(len(plan["files"]), 1)
        self.assertEqual(plan["freed"], 4)
        result = remove.remove(profile, root=self.root, apply=True)
        self.assertFalse(link.exists())
        self.assertFalse(blob.exists())
        self.assertEqual(result["freed"], 4)

    def test_apply_builtin_deletes_files_and_keeps_profile(self):
        builtin = config.BUILTIN_PROFILES["glm-4.7-flash-ud-q4-k-xl"]
        model_id = builtin["id"]
        data = b"gguf"
        config.set_context(model_id, 65536)
        link, blob = _link(builtin["repo"], builtin["filename"],
                           builtin["revision"], self.root, data)
        config.reload_profiles()
        profile = config.PROFILES[model_id]
        self.assertEqual(profile["path"], str(link))

        result = remove.remove(profile, root=self.root, apply=True)
        self.assertFalse(link.exists())
        self.assertFalse(blob.exists())
        self.assertIn(model_id, config.PROFILES)
        self.assertFalse(Path(config.PROFILES[model_id]["path"]).exists())
        registry = config.load_registry(self.models)
        self.assertNotIn(model_id, registry.overrides)

    def test_apply_user_model_deletes_entry_and_override(self):
        repo = "org/model"
        link, blob = _link(repo, "mine.gguf", COMMIT, self.root, b"gguf")
        other_link, _other_blob = _link(repo, "other.gguf", COMMIT,
                                        self.root, b"other")
        self.models.write_text(json.dumps({
            "models": {
                "mine": _profile("mine", "mine", repo, "mine.gguf", "main", link),
                "other": _profile("other", "other", repo, "other.gguf",
                                  "main", other_link),
            },
            "overrides": {"mine": {"context": 65536}},
        }))
        profile = _profile("mine", "mine", repo, "mine.gguf", "main", link)
        result = remove.remove(profile, root=self.root, apply=True)
        self.assertFalse(link.exists())
        registry = config.load_registry(self.models)
        self.assertNotIn("mine", registry.models)
        self.assertNotIn("mine", registry.overrides)
        self.assertIn("other", registry.models)

    def test_apply_stops_the_running_model(self):
        repo = "org/model"
        link, blob = _link(repo, "mine.gguf", COMMIT, self.root, b"gguf")
        profile = _profile("mine", "mine", repo, "mine.gguf", "main", link,
                           source="cache")
        with mock.patch.object(service, "running", return_value=123), \
             mock.patch.object(service, "identity",
                               return_value="llama-server --alias mine --port 18081"), \
             mock.patch.object(service, "stop") as stop:
            result = remove.remove(profile, root=self.root, apply=True)
        stop.assert_called_once()
        self.assertFalse(link.exists())
        self.assertFalse(blob.exists())

    def test_plan_refuses_path_outside_cache(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        victim = Path(outside.name) / "snapshots" / COMMIT / "victim.gguf"
        victim.parent.mkdir(parents=True, exist_ok=True)
        victim.write_bytes(b"do not delete")
        profile = _profile("m", "m", "org/model", "victim.gguf", "main", victim)
        with self.assertRaises(config.ModelError):
            remove.remove(profile, root=self.root, apply=True)
        self.assertTrue(victim.exists())

    def test_plan_refuses_shard_escaping_the_cache(self):
        repo = "org/model"
        links, blobs = _split_links(repo, COMMIT, self.root)
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        victim = Path(outside.name) / "victim.gguf"
        victim.write_bytes(b"do not delete")
        shard2 = links[1]
        shard2.unlink()
        os.symlink(os.path.relpath(victim, shard2.parent), shard2)
        profile = _profile("m-q4", "q4", repo, "q4/M-Q4-00001-of-00002.gguf",
                           "main", links[0])
        with self.assertRaises(config.ModelError):
            remove.remove(profile, root=self.root, apply=True)
        self.assertTrue(victim.exists())
        self.assertTrue(blobs[0].exists())

    def test_plan_refuses_unsafe_revision(self):
        repo = "org/model"
        link, blob = _link(repo, "m.gguf", COMMIT, self.root, b"gguf")
        profile = _profile("m", "m", repo, "m.gguf", "../evil", link)
        with self.assertRaises(config.ModelError):
            remove.remove(profile, root=self.root, apply=True)
        self.assertTrue(link.exists())
        self.assertTrue(blob.exists())

    def test_split_set_does_not_over_collect_shared_stem(self):
        repo = "org/model"
        links, _blobs = _split_links(repo, COMMIT, self.root, stem="M-Q4")
        _split_links(repo, COMMIT, self.root, stem="M-Q4-K")
        profile = _profile("m-q4", "q4", repo, "q4/M-Q4-00001-of-00002.gguf",
                           "main", links[0])
        result = remove.plan(profile, root=self.root)
        filenames = sorted(item["filename"] for item in result["files"])
        self.assertEqual(filenames, [
            "q4/M-Q4-00001-of-00002.gguf",
            "q4/M-Q4-00002-of-00002.gguf",
        ])
        self.assertEqual(len(result["files"]), 2)



if __name__ == "__main__":
    unittest.main()

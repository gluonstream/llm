import os
import tempfile
import unittest
from pathlib import Path

from llm import hfcache

COMMIT = "a" * 40
OLD_COMMIT = "b" * 40
OTHER_COMMIT = "c" * 40


class HFCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def repo(self, repo="org/model"):
        return self.root / ("models--" + repo.replace("/", "--"))

    def snapshot(self, repo="org/model", commit=COMMIT):
        return self.repo(repo) / "snapshots" / commit

    def write(self, path, data=b"x"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_cache_root_precedence(self):
        first = self.root / "first"
        second = self.root / "second"
        hf_home = self.root / "hf-home"
        self.assertEqual(
            hfcache.cache_root({
                "HF_HUB_CACHE": str(first),
                "HUGGINGFACE_HUB_CACHE": str(second),
                "HF_HOME": str(hf_home),
            }),
            first.resolve())
        self.assertEqual(
            hfcache.cache_root({
                "HUGGINGFACE_HUB_CACHE": str(second),
                "HF_HOME": str(hf_home),
            }),
            second.resolve())
        self.assertEqual(
            hfcache.cache_root({"HF_HOME": str(hf_home)}),
            (hf_home / "hub").resolve())
        self.assertEqual(
            hfcache.cache_root({}),
            Path.home() / ".cache" / "huggingface" / "hub")
        self.assertEqual(
            hfcache.cache_root({"HF_HUB_CACHE": "~/hf-cache"}),
            (Path.home() / "hf-cache").resolve())

    def test_read_ref_and_expected_snapshot(self):
        refs = self.repo() / "refs"
        self.write(refs / "main", COMMIT.encode())
        self.write(refs / "empty", b"")
        self.assertEqual(hfcache.read_ref("org/model", "main", self.root), COMMIT)
        self.assertIsNone(hfcache.read_ref("org/model", "dev", self.root))
        self.assertIsNone(hfcache.read_ref("org/model", "empty", self.root))

        self.assertEqual(
            hfcache.expected_snapshot("org/model", "q4/file.gguf", "main",
                                      self.root),
            self.repo() / "snapshots" / COMMIT / "q4" / "file.gguf")
        self.assertEqual(
            hfcache.expected_snapshot("org/model", "x.gguf", OLD_COMMIT,
                                      self.root),
            self.repo() / "snapshots" / OLD_COMMIT / "x.gguf")
        self.assertEqual(
            hfcache.expected_snapshot("org/model", "x.gguf", "dev", self.root),
            self.repo() / "snapshots" / "dev" / "x.gguf")

    def test_split_set_parsing(self):
        self.assertEqual(hfcache.split_set("x-00001-of-00002.gguf"),
                         ("x", 1, 2))
        self.assertEqual(hfcache.split_set("Model-00012-of-00013.GGUF"),
                         ("Model", 12, 13))
        self.assertEqual(
            hfcache.split_set(Path("nested/x-00003-of-00004.gguf")),
            ("x", 3, 4))
        self.assertIsNone(hfcache.split_set("plain.gguf"))
        self.assertIsNone(hfcache.split_set("x-00001-of-00002-extra.gguf"))

    def test_group_ggufs_complete_and_incomplete_split(self):
        folder = self.root / "files"
        one = self.write(folder / "a-00001-of-00002.gguf", b"abc")
        two = self.write(folder / "a-00002-of-00002.gguf", b"12345")
        groups = hfcache.group_ggufs([two, one])
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["entry"], one)
        self.assertEqual(group["files"], [one, two])
        self.assertTrue(group["split"])
        self.assertEqual(group["size"], 8)

        first_incomplete = self.write(folder / "b-00001-of-00003.gguf", b"x")
        self.write(folder / "b-00003-of-00003.gguf", b"x")
        self.assertEqual(hfcache.group_ggufs([first_incomplete]), [])

        plain = self.write(folder / "plain.gguf", b"xyz")
        groups = hfcache.group_ggufs([plain])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["entry"], plain)
        self.assertEqual(groups[0]["files"], [plain])
        self.assertFalse(groups[0]["split"])
        self.assertEqual(groups[0]["size"], 3)

        empty = self.write(folder / "zero.gguf", b"")
        self.assertEqual(hfcache.group_ggufs([empty])[0]["size"], 0)

    def test_discover_prefers_ref_main_and_dedupes(self):
        repo = "org/model"
        self.write(self.repo(repo) / "refs" / "main", COMMIT.encode())
        main = self.write(self.snapshot(repo) / "q4" / "file.gguf", b"new")
        self.write(self.repo(repo) / "snapshots" / OLD_COMMIT
                   / "q4" / "file.gguf", b"old")
        extra = self.write(self.repo(repo) / "snapshots" / OLD_COMMIT
                           / "only-old.gguf", b"extra")

        found = hfcache.discover(root=self.root)
        self.assertEqual(len(found), 2)
        by_filename = {profile["filename"]: profile
                       for profile in found.values()}
        self.assertEqual(set(by_filename), {"q4/file.gguf", "only-old.gguf"})
        self.assertEqual(by_filename["q4/file.gguf"]["path"], str(main))
        self.assertEqual(by_filename["q4/file.gguf"]["revision"], "main")
        self.assertEqual(by_filename["only-old.gguf"]["path"], str(extra))
        self.assertEqual(by_filename["only-old.gguf"]["revision"], OLD_COMMIT)

    def test_discover_skips_mmproj_part_and_broken(self):
        snapshot = self.snapshot()
        self.write(snapshot / "mmproj-x.gguf", b"x")
        self.write(snapshot / "mtp-extra.gguf", b"x")
        self.write(snapshot / "y.gguf.part", b"x")
        self.write(snapshot / ".hidden.gguf", b"x")
        self.write(snapshot / "empty.gguf", b"")
        os.symlink("missing-target.gguf", snapshot / "broken.gguf")

        self.assertEqual(hfcache.discover(root=self.root), {})
        self.assertEqual(hfcache.discover(root=self.root / "missing"), {})

        good = self.write(snapshot / "good.gguf", b"ok")
        found = hfcache.discover(root=self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(list(found.values())[0]["path"], str(good))

    def test_discover_synthesizes_deterministic_ids_and_aliases(self):
        repo = "org/Model-Repo"
        snapshot = self.snapshot(repo)
        self.write(snapshot / "q4" / "Split.Model-00001-of-00002.gguf", b"a")
        self.write(snapshot / "q4" / "Split.Model-00002-of-00002.gguf", b"bc")

        first = hfcache.discover(root=self.root)
        second = hfcache.discover(root=self.root)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        profile = list(first.values())[0]
        self.assertEqual(profile["alias"], "split-model")
        self.assertEqual(
            profile["id"],
            "org-model-repo-q4-split-model-00001-of-00002")
        self.assertEqual(profile["filename"],
                         "q4/Split.Model-00001-of-00002.gguf")
        self.assertTrue(profile["path"].endswith(
            "Split.Model-00001-of-00002.gguf"))

    def test_discover_dedupes_colliding_aliases(self):
        for repo in ("org/one", "org/two"):
            self.write(self.snapshot(repo) / "same.gguf", b"x")

        found = hfcache.discover(root=self.root)
        self.assertEqual(len(found), 2)
        self.assertEqual(sorted(profile["alias"]
                                for profile in found.values()),
                         ["same", "same-2"])

    def test_discover_handles_empty_slugs(self):
        snapshot = self.snapshot()
        self.write(snapshot / "---.gguf", b"x")
        self.write(snapshot / "+++.gguf", b"x")

        found = hfcache.discover(root=self.root)
        self.assertEqual(len(found), 2)
        self.assertEqual(sorted(profile["alias"]
                                for profile in found.values()),
                         ["model", "model-2"])

    def test_profile_for_shape(self):
        repo = "org/model"
        path = self.write(self.snapshot(repo) / "q4" / "file.gguf", b"abc")
        group = hfcache.group_ggufs([path])[0]

        profile = hfcache.profile_for(repo, "main", group, root=self.root)
        self.assertEqual(profile["id"], "org-model-q4-file")
        self.assertEqual(profile["alias"], "file")
        self.assertEqual(profile["repo"], repo)
        self.assertEqual(profile["filename"], "q4/file.gguf")
        self.assertEqual(profile["revision"], "main")
        self.assertEqual(profile["path"], str(path))
        self.assertEqual(profile["context"], 8192)
        self.assertEqual(profile["sampling"], hfcache.DEFAULT_SAMPLING)
        self.assertEqual(profile["source"], "cache")

        profile["sampling"]["temperature"] = 99
        self.assertEqual(hfcache.DEFAULT_SAMPLING["temperature"], 0.7)

        custom = hfcache.profile_for(repo, "main", group, root=self.root,
                                     sampling={"temperature": 0})
        self.assertEqual(custom["sampling"], {"temperature": 0})

    def test_resolve_snapshot_prefers_expected_then_newest(self):
        repo = "org/model"
        self.write(self.repo(repo) / "refs" / "main", COMMIT.encode())
        expected = self.write(self.snapshot(repo) / "file.gguf", b"main")
        older = self.write(self.repo(repo) / "snapshots" / OLD_COMMIT
                           / "file.gguf", b"old")
        self.assertEqual(
            hfcache.resolve_snapshot(repo, "file.gguf", "main", self.root),
            expected)

        self.write(self.repo(repo) / "refs" / "dev", OTHER_COMMIT.encode())
        os.utime(self.snapshot(repo), (1, 1))
        os.utime(self.repo(repo) / "snapshots" / OLD_COMMIT, (2, 2))
        self.assertEqual(
            hfcache.resolve_snapshot(repo, "file.gguf", "dev", self.root),
            older)
        self.assertIsNone(
            hfcache.resolve_snapshot(repo, "absent.gguf", "main", self.root))


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import hfcache, migrate

COMMIT = "a" * 40
PINNED = "b" * 40


def digest(data):
    return hashlib.sha256(data).hexdigest()


class MigrateTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.cache = self.temp / "hub"
        self.legacy = self.temp / "share"

    def write(self, path, data=b"x"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def entry(self, files, repo="org/model", commit=COMMIT, revision="main",
              source_dir=None):
        source_dir = Path(source_dir or (self.legacy / "graphify-model"))
        return {
            "source": str(source_dir),
            "repo": repo,
            "commit": commit,
            "revision": revision,
            "origin": "known",
            "files": [
                {"filename": name, "sha256": digest(data), "size": len(data),
                 "path": str(source_dir / name)}
                for name, data in files.items()
            ],
        }

    def repo(self, repo="org/model"):
        return hfcache.repo_dir(repo, self.cache)

    def test_known_mapping_and_manifest_scan(self):
        for mapping in migrate.KNOWN_MIGRATIONS:
            source = self.legacy / Path(mapping["source"]).name
            for item in mapping["files"]:
                self.write(source / item["filename"])
            # A manifest in a known directory must not duplicate it.
            (source / "download-manifest.json").write_text(json.dumps({
                "url": "https://huggingface.co/ignored/duplicate/resolve/main/x.gguf",
                "revision": "main", "bytes": 1, "sha256": "0" * 64,
            }))

        extra = self.legacy / "graphify-extra"
        payload = b"extra model bytes"
        self.write(extra / "Extra-Q4_K_M.gguf", payload)
        (extra / "download-manifest.json").write_text(json.dumps({
            "url": ("https://huggingface.co/org/extra/resolve/main/"
                    "Extra-Q4_K_M.gguf"),
            "revision": "main",
            "bytes": len(payload),
            "sha256": digest(payload),
        }))

        entries = migrate.plan(legacy_root=self.legacy, root=self.cache)
        self.assertEqual(len(entries), len(migrate.KNOWN_MIGRATIONS) + 1)
        by_repo = {entry["repo"]: entry for entry in entries}

        qwen = by_repo["Qwen/Qwen2.5-7B-Instruct-GGUF"]
        self.assertEqual(qwen["commit"],
                         "bb5d59e06d9551d752d08b292a50eb208b07ab1f")
        self.assertEqual(qwen["revision"], "main")
        self.assertEqual(qwen["status"], "ready")
        self.assertEqual(len(qwen["files"]), 2)
        self.assertEqual(qwen["files"][0]["size"], 3993201344)
        self.assertTrue(qwen["files"][0]["present"])

        extra_entry = by_repo["org/extra"]
        self.assertEqual(extra_entry["commit"], "main")
        self.assertEqual(extra_entry["revision"], "main")
        self.assertEqual(extra_entry["origin"], "manifest")
        self.assertEqual(extra_entry["status"], "ready")
        self.assertEqual(len(extra_entry["files"]), 1)
        item = extra_entry["files"][0]
        self.assertEqual(item["filename"], "Extra-Q4_K_M.gguf")
        self.assertEqual(item["sha256"], digest(payload))
        self.assertEqual(item["size"], len(payload))

    def test_plan_marks_missing_sources(self):
        entries = migrate.plan(legacy_root=self.legacy, root=self.cache)
        self.assertEqual(len(entries), len(migrate.KNOWN_MIGRATIONS))
        self.assertTrue(all(entry["status"] == "missing" for entry in entries))

    def test_migrate_hardlinks_and_writes_layout(self):
        payload = b"hello world"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        inode = source.stat().st_ino
        entry = self.entry({"model.gguf": payload})

        result = migrate.migrate(entry, root=self.cache)

        self.assertEqual(result["status"], "migrated")
        self.assertEqual(result["files"][0]["status"], "migrated")
        self.assertEqual(result["files"][0]["method"], "hardlink")

        blob = self.repo() / "blobs" / digest(payload)
        self.assertEqual(blob.read_bytes(), payload)
        self.assertEqual(blob.stat().st_ino, inode)

        snapshot = self.repo() / "snapshots" / COMMIT / "model.gguf"
        self.assertTrue(snapshot.is_symlink())
        self.assertEqual(os.readlink(snapshot),
                         "../../blobs/" + digest(payload))
        self.assertEqual(snapshot.read_bytes(), payload)
        self.assertEqual(hfcache.read_ref("org/model", "main", self.cache),
                         COMMIT)

        self.assertFalse((self.legacy / "graphify-model").exists())

    def test_pinned_commit_writes_no_ref(self):
        payload = b"pinned"
        self.write(self.legacy / "graphify-model" / "model.gguf", payload)
        entry = self.entry({"model.gguf": payload}, commit=PINNED,
                           revision=PINNED)

        result = migrate.migrate(entry, root=self.cache)

        self.assertEqual(result["status"], "migrated")
        self.assertIsNone(hfcache.read_ref("org/model", "main", self.cache))
        snapshot = self.repo() / "snapshots" / PINNED / "model.gguf"
        self.assertEqual(snapshot.read_bytes(), payload)

    def test_migrate_copy_fallback(self):
        payload = b"copy me across filesystems"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        entry = self.entry({"model.gguf": payload})

        with mock.patch.object(migrate.os, "link",
                               side_effect=OSError("cross-device")):
            result = migrate.migrate(entry, root=self.cache)

        self.assertEqual(result["status"], "migrated")
        self.assertEqual(result["files"][0]["method"], "copy")
        blob = self.repo() / "blobs" / digest(payload)
        self.assertEqual(blob.read_bytes(), payload)
        self.assertFalse(source.exists())
        self.assertFalse((self.legacy / "graphify-model").exists())

    def test_migrate_is_idempotent(self):
        payload = b"same bytes"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        entry = self.entry({"model.gguf": payload})

        first = migrate.migrate(entry, root=self.cache, keep_old=True)
        second = migrate.migrate(entry, root=self.cache, keep_old=True)

        self.assertEqual(first["status"], "migrated")
        self.assertEqual(second["status"], "already present")
        self.assertEqual(second["files"][0]["status"], "already present")
        self.assertEqual(second["files"][0]["method"], "reused")
        self.assertEqual(len(list((self.repo() / "blobs").iterdir())), 1)
        self.assertTrue(source.exists())
        self.assertEqual(
            (self.repo() / "snapshots" / COMMIT / "model.gguf").read_bytes(),
            payload)

    def test_migrate_dry_run_changes_nothing(self):
        payload = b"dry run bytes"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        entry = self.entry({"model.gguf": payload})

        result = migrate.migrate(entry, root=self.cache, dry_run=True)

        self.assertEqual(result["status"], "dry-run")
        self.assertEqual(result["files"][0]["status"], "would migrate")
        self.assertTrue(source.exists())
        self.assertFalse(self.repo().exists())
        self.assertFalse(self.cache.exists())

    def test_migrate_verify_detects_bad_sha(self):
        payload = b"corrupted source"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        entry = self.entry({"model.gguf": payload})
        entry["files"][0]["sha256"] = "0" * 64

        result = migrate.migrate(entry, root=self.cache, verify=True)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["files"][0]["status"], "error")
        self.assertIn("checksum", result["files"][0]["message"].lower())
        self.assertTrue(source.exists())
        self.assertFalse(self.cache.exists())

    def test_migrate_rejects_wrong_source_size(self):
        payload = b"incomplete download"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        entry = self.entry({"model.gguf": payload})
        entry["files"][0]["size"] = len(payload) + 1

        result = migrate.migrate(entry, root=self.cache)

        self.assertEqual(result["status"], "error")
        self.assertIn("expected", result["files"][0]["message"])
        self.assertTrue(source.exists())
        self.assertFalse(self.cache.exists())

    def test_migrate_keeps_old_directory_with_unrelated_files(self):
        payload = b"model"
        source = self.write(self.legacy / "graphify-model" / "model.gguf",
                            payload)
        notes = self.write(self.legacy / "graphify-model" / "notes.txt", b"hi")
        entry = self.entry({"model.gguf": payload})

        result = migrate.migrate(entry, root=self.cache)

        self.assertEqual(result["status"], "migrated")
        self.assertTrue(source.exists())
        self.assertTrue(notes.exists())
        self.assertEqual([Path(path).name for path in result["leftover"]],
                         ["notes.txt"])


if __name__ == "__main__":
    unittest.main()

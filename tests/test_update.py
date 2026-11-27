import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import update


class FakeRunner:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, command, cwd=None, check=False):
        self.calls.append((list(command), cwd))
        if self.fail_on is not None and list(command) == list(self.fail_on):
            raise subprocess.CalledProcessError(1, command)


class UpdateTest(unittest.TestCase):
    def test_temp_root_is_os_specific(self):
        self.assertEqual(update.temp_root("darwin"), Path("/private/tmp"))
        self.assertEqual(update.temp_root("linux"), Path("/tmp"))

    def test_default_checkout_honors_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom"
            with mock.patch.dict(os.environ, {"LMM_UPDATE_DIR": str(target)}):
                self.assertEqual(update.default_checkout(), target)

    def test_update_clones_then_builds_and_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            runner = FakeRunner()
            result = update.update(checkout=checkout, runner=runner)
        self.assertEqual(result, checkout)
        self.assertEqual(runner.calls, [
            (["git", "clone", update.REPO_URL, str(checkout)], None),
            (["./build.sh"], str(checkout)),
            (["./install.sh"], str(checkout)),
        ])

    def test_update_pulls_existing_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            (checkout / ".git").mkdir(parents=True)
            runner = FakeRunner()
            update.update(checkout=checkout, runner=runner)
        self.assertEqual(runner.calls[0],
                         (["git", "-C", str(checkout), "pull", "--ff-only"], None))

    def test_update_refuses_non_git_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            checkout.mkdir()
            runner = FakeRunner()
            with self.assertRaises(update.UpdateError):
                update.update(checkout=checkout, runner=runner)
        self.assertEqual(runner.calls, [])

    def test_update_reports_failed_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            runner = FakeRunner(fail_on=["git", "clone", update.REPO_URL, str(checkout)])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("git clone", str(caught.exception))

    def test_update_reports_failed_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            (checkout / ".git").mkdir(parents=True)
            runner = FakeRunner(
                fail_on=["git", "-C", str(checkout), "pull", "--ff-only"])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("pull --ff-only", str(caught.exception))

    def test_update_reports_failed_build_and_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            runner = FakeRunner(fail_on=["./build.sh"])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("./build.sh", str(caught.exception))
        self.assertNotIn(("./install.sh"), [call[0][0] for call in runner.calls])

    def test_update_reports_missing_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(update.shutil, "which", return_value=None):
                with self.assertRaises(update.UpdateError) as caught:
                    update.update(checkout=Path(tmp) / "llm-update",
                                  runner=FakeRunner())
        self.assertIn("git", str(caught.exception))

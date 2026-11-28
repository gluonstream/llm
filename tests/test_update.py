import base64
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from llm import update


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.text.encode("utf-8")


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

    def test_update_rejects_unsupported_os_before_any_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            with mock.patch.object(update.host, "IS_MACOS", False), \
                 mock.patch.object(update.host, "IS_LINUX", False):
                with self.assertRaises(update.UpdateError) as caught:
                    update.update(checkout=Path(tmp) / "llm-update", runner=runner)
        self.assertIn("unsupported OS", str(caught.exception))
        self.assertEqual(runner.calls, [])

    def test_latest_version_reads_last_version_file(self):
        def opener(url, timeout):
            self.assertEqual(url, update.LAST_VERSION_URL)
            return FakeResponse("1.4.0\n")
        self.assertEqual(update.latest_version(opener=opener), "1.4.0")

    def test_latest_version_decodes_github_api_json(self):
        payload = json.dumps({
            "content": base64.b64encode(b"1.4.6\n").decode("ascii")})
        self.assertEqual(
            update.latest_version(
                opener=lambda url, timeout: FakeResponse(payload)),
            "1.4.6")

    def test_latest_version_returns_none_on_failure_or_invalid(self):
        def opener(url, timeout):
            raise OSError("offline")
        self.assertIsNone(update.latest_version(opener=opener))
        self.assertIsNone(update.latest_version(
            opener=lambda url, timeout: FakeResponse("not a version")))

    def test_needs_update_compares_versions(self):
        self.assertTrue(update.needs_update("1.4.1", "1.4.0"))
        self.assertFalse(update.needs_update("1.4.0", "1.4.0"))
        self.assertFalse(update.needs_update("1.3.9", "1.4.0"))
        self.assertFalse(update.needs_update("nope", "1.4.0"))

    def test_main_skips_when_up_to_date(self):
        buffer = io.StringIO()
        with mock.patch.object(update, "latest_version", return_value="1.4.0"), \
             mock.patch.object(update, "update") as patched, \
             redirect_stdout(buffer):
            self.assertEqual(update.main([]), 0)
        patched.assert_not_called()
        self.assertIn("up to date", buffer.getvalue())

    def test_main_force_updates(self):
        with mock.patch.object(update, "latest_version", return_value="1.4.0"), \
             mock.patch.object(update, "update", return_value=Path("/tmp/x")) as patched, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(update.main(["--force"]), 0)
        patched.assert_called_once()

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
            (["git", "clone", "--branch", "main", update.REPO_URL, str(checkout)], None),
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
                         (["git", "-C", str(checkout), "fetch", "origin", "main"], None))
        self.assertEqual(runner.calls[1],
                         (["git", "-C", str(checkout), "checkout", "-f", "-B",
                           "main", "origin/main"], None))

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
            runner = FakeRunner(fail_on=["git", "clone", "--branch", "main", update.REPO_URL, str(checkout)])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("git clone", str(caught.exception))

    def test_update_reports_failed_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            (checkout / ".git").mkdir(parents=True)
            runner = FakeRunner(
                fail_on=["git", "-C", str(checkout), "fetch", "origin", "main"])
            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("fetch", str(caught.exception))

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

    def test_update_expands_user_in_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            with mock.patch.dict(os.environ, {"HOME": tmp}):
                result = update.update(checkout="~/custom", runner=runner)
            expected = Path(tmp) / "custom"
        self.assertEqual(result, expected)
        self.assertEqual(runner.calls[0][0],
                         ["git", "clone", "--branch", "main", update.REPO_URL, str(expected)])

    def test_main_forwards_dir_to_update(self):
        with mock.patch.object(update, "latest_version", return_value=None), \
             mock.patch.object(update, "update",
                               return_value=Path("/tmp/custom")) as patched:
            with mock.patch("builtins.print"):
                self.assertEqual(update.main(["--dir", "/tmp/custom"]), 0)
        patched.assert_called_once_with(checkout="/tmp/custom")

    def test_update_reports_unlaunchable_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "llm-update"
            (checkout / ".git").mkdir(parents=True)

            def runner(command, cwd=None, check=False):
                if command == ["./build.sh"]:
                    raise PermissionError(13, "Permission denied")

            with self.assertRaises(update.UpdateError) as caught:
                update.update(checkout=checkout, runner=runner)
        self.assertIn("./build.sh", str(caught.exception))
        self.assertIn("Permission denied", str(caught.exception))

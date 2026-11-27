import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build.sh"


def env_for(tmp, **extra):
    env = dict(os.environ)
    state = Path(tmp) / "state"
    state.mkdir(exist_ok=True)
    env.update({
        "LMM_STATE_DIR": str(state),
        "LMM_LEGACY_STATE_DIR": str(Path(tmp) / "legacy-missing"),
        "LMM_BUILD_DIR": str(Path(tmp) / "build dir"),
        "LMM_DIST_DIR": str(Path(tmp) / "dist dir"),
        "LMM_SKIP_TESTS": "1",
        "LMM_SKIP_LLAMA_CPP_UPGRADE": "1",
    })
    env.update(extra)
    return env


class BuildTest(unittest.TestCase):
    def test_build_produces_executable_zipapp_with_bytecode_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                    env=env_for(tmp), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = Path(tmp) / "dist dir" / "llm"
            self.assertTrue(artifact.is_file())
            self.assertTrue(os.access(artifact, os.X_OK))
            self.assertTrue(zipfile.is_zipfile(artifact))
            with zipfile.ZipFile(artifact) as archive:
                names = set(archive.namelist())
            for expected in ("__main__.py", "llm/cli.pyc",
                             "llm/cli.py"):
                self.assertIn(expected, names)

    def test_build_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = env_for(tmp)
            for _ in range(2):
                result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "dist dir" / "llm").is_file())

    def test_build_runs_unit_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            tests_dir = Path(tmp) / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_ok.py").write_text(
                "import unittest\n\n"
                "class Ok(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        pass\n")
            env = env_for(tmp, LMM_TEST_DIR=str(tests_dir))
            env.pop("LMM_SKIP_TESTS")
            result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Ran 1 test", result.stdout + result.stderr)
            self.assertTrue((Path(tmp) / "dist dir" / "llm").is_file())

    def test_build_stops_when_unit_tests_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tests_dir = Path(tmp) / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_fail.py").write_text(
                "import unittest\n\n"
                "class Fail(unittest.TestCase):\n"
                "    def test_fail(self):\n"
                "        self.fail('boom')\n")
            env = env_for(tmp, LMM_TEST_DIR=str(tests_dir))
            env.pop("LMM_SKIP_TESTS")
            result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAILED", result.stdout + result.stderr)
            self.assertFalse((Path(tmp) / "dist dir" / "llm").exists())

    def test_build_rejects_unsupported_os(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(BUILD)], cwd=ROOT,
                env=env_for(tmp, OSTYPE="plan9"),
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported OS", result.stderr)

    def test_build_skips_package_check_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp) / "bin"
            bindir.mkdir()
            marker = Path(tmp) / "brew-called"
            brew = bindir / "brew"
            brew.write_text("#!/bin/sh\ntouch %s\n" % marker)
            brew.chmod(0o755)
            env = env_for(tmp)
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(["bash", str(BUILD)], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.env = env_for(self.tmp.name,
                           LMM_BIN_DIR=str(self.tmp_path / "bin dir"))
        self.bin_dir = self.tmp_path / "bin dir"
        built = subprocess.run(["bash", str(ROOT / "build.sh")], cwd=ROOT,
                               env=self.env, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)

    def install(self):
        return subprocess.run(["bash", str(ROOT / "install.sh")], cwd=ROOT,
                              env=self.env, capture_output=True, text=True)

    def test_installs_llm_and_relative_aliases(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        llm = self.bin_dir / "llm"
        self.assertTrue(llm.is_file())
        self.assertTrue(os.access(llm, os.X_OK))
        for alias in ("lmm",):
            link = self.bin_dir / alias
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "llm")
        self.assertEqual(self.run_help(llm), 0)

    def test_replaces_existing_regular_file(self):
        self.bin_dir.mkdir(parents=True)
        (self.bin_dir / "llm").write_text("old")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(zipfile.is_zipfile(self.bin_dir / "llm"))
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)

    def test_replaces_existing_symlink(self):
        self.bin_dir.mkdir(parents=True)
        (self.bin_dir / "llm").symlink_to("/dev/null")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.bin_dir / "llm").is_symlink())
        self.assertTrue(zipfile.is_zipfile(self.bin_dir / "llm"))
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)

    def test_replaces_symlink_to_directory(self):
        target = self.tmp_path / "real dir"
        target.mkdir()
        self.bin_dir.mkdir(parents=True)
        (self.bin_dir / "llm").symlink_to(target)
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.bin_dir / "llm").is_symlink())
        self.assertTrue(zipfile.is_zipfile(self.bin_dir / "llm"))
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)
        self.assertEqual(list(target.iterdir()), [])

    def test_refuses_directory_at_llm(self):
        (self.bin_dir / "llm").mkdir(parents=True)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("directory", result.stderr)
        self.assertTrue((self.bin_dir / "llm").is_dir())

    def test_refuses_directory_alias(self):
        self.bin_dir.mkdir(parents=True)
        (self.bin_dir / "lmm").mkdir()
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("directory", result.stderr)
        self.assertTrue((self.bin_dir / "lmm").is_dir())

    def test_install_is_idempotent(self):
        for _ in range(2):
            self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.run_help(self.bin_dir / "llm"), 0)

    def test_missing_artifact_names_build_script(self):
        empty = self.tmp_path / "empty dist"
        env = dict(self.env, LMM_DIST_DIR=str(empty))
        result = subprocess.run(["bash", str(ROOT / "install.sh")], cwd=ROOT,
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("./build.sh", result.stderr)

    def test_install_rejects_unsupported_os(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(ROOT / "install.sh")], cwd=ROOT,
                env=env_for(tmp, OSTYPE="plan9"),
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported OS", result.stderr)

    def run_help(self, executable):
        return subprocess.run([str(executable), "help"], env=self.env,
                              capture_output=True, text=True).returncode


class UpdateScriptTest(unittest.TestCase):
    def test_update_script_is_executable_and_rejects_unsupported_os(self):
        script = ROOT / "update.sh"
        self.assertTrue(os.access(script, os.X_OK))
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(script)], cwd=ROOT,
                env=env_for(tmp, OSTYPE="plan9"),
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported OS", result.stderr)

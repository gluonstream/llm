import unittest
from unittest import mock

from llm import diagnose

class DiagnoseTest(unittest.TestCase):
    def test_parse_and_compare(self):
        self.assertEqual(diagnose.parse_version("version: 1.0.0 (build 12359, commit b1b)"),
                         (1, 0, 0))
        self.assertTrue(diagnose.version_at_least("1.1.0", "1.1.0"))
        self.assertFalse(diagnose.version_at_least("1.0.0", "1.1.0"))

    def test_unknown_architecture_inkling(self):
        log = ("E llama_model_load: error loading model: "
               "unknown model architecture: 'inkling2_loop'\n")
        message = diagnose.explain_failure(log, upgrade_hint="brew upgrade llama.cpp")
        self.assertIn("inkling2_loop", message)
        self.assertIn("1.1.0", message)
        self.assertIn("brew upgrade llama.cpp", message)

    def test_generic_load_failure_mentions_logs(self):
        log = "E srv load_model: failed to load model, '/x/y.gguf'\n"
        message = diagnose.explain_failure(log)
        self.assertIn("failed to load model", message)
        self.assertIn("lmm logs", message)

    def test_clean_log_has_no_explanation(self):
        self.assertIsNone(diagnose.explain_failure("all good\n"))

    def test_explain_failure_accepts_an_upgrade_hint(self):
        log = "error loading model: unknown model architecture: 'inkling2_loop'\n"
        self.assertIn("custom hint", diagnose.explain_failure(log, upgrade_hint="custom hint"))

    def test_llama_server_version_uses_config_path(self):
        with mock.patch.object(diagnose.config, "LLAMA_SERVER", "/custom/llama-server"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "version: 1.1.0"
            diagnose.llama_server_version()
        self.assertEqual(run.call_args[0][0][0], "/custom/llama-server")

    def test_arch_package_status_uses_pacman_owner(self):
        with mock.patch.object(diagnose.host, "IS_MACOS", False), \
             mock.patch.object(diagnose.host, "IS_LINUX", True), \
             mock.patch.object(diagnose.config, "LLAMA_SERVER", "/usr/bin/llama-server"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = (
                "/usr/bin/llama-server is owned by llama.cpp-git r1-1\n")
            self.assertIn("llama.cpp-git", diagnose.llama_cpp_package_status())

    def test_macos_package_status_uses_brew(self):
        with mock.patch.object(diagnose.host, "IS_MACOS", True), \
             mock.patch.object(diagnose, "brew_llama_cpp_status",
                               return_value="Homebrew llama.cpp 1.1.0"):
            self.assertEqual(diagnose.llama_cpp_package_status(),
                             "Homebrew llama.cpp 1.1.0")

import unittest
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
        message = diagnose.explain_failure(log)
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

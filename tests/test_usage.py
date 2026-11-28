import tempfile
import unittest
from pathlib import Path

from llm import usage


class UsageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write(self, text):
        path = self.dir / "server.log"
        path.write_text(text)
        return path

    def test_peak_uses_last_model_load(self):
        path = self._write(
            "loading model '/models/a.gguf'\n"
            "slot release: n_tokens = 5000\n"
            "loading model '/models/b.gguf'\n"
            "slot release: n_tokens = 4000\n"
            "prompt processing: n_tokens = 3000\n")
        self.assertEqual(
            usage.peak_tokens(path, model_path="/models/b.gguf"), 4000)

    def test_peak_reads_release_and_prompt_lines(self):
        path = self._write(
            "slot release: n_tokens = 1200\n"
            "prompt eval time =   300.00 ms /  2048 tokens "
            "(   0.15 ms per token,  6826.67 tokens per second)\n")
        self.assertEqual(usage.peak_tokens(path), 2048)

    def test_peak_ignores_zero_and_missing_log(self):
        path = self._write("slot release: n_tokens = 0\n")
        self.assertIsNone(usage.peak_tokens(path))
        self.assertIsNone(usage.peak_tokens(self.dir / "missing.log"))


if __name__ == "__main__":
    unittest.main()

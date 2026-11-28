import unittest
from llm import render

PROFILES = {
    "inkling-2-mini-gguf-ud-q4-k-xl": {"id": "inkling-2-mini-gguf-ud-q4-k-xl", "alias": "inkling-2-mini"},
    "qwen2.5-7b-instruct-q4-k-m": {"id": "qwen2.5-7b-instruct-q4-k-m", "alias": "qwen2.5-7b-instruct"},
}

class RenderTest(unittest.TestCase):
    def test_human_size(self):
        self.assertEqual(render.human_size(17629760736), "16.4 GiB")

    def test_list_marks_selected_and_running(self):
        text = render.format_list(
            PROFILES, "inkling-2-mini-gguf-ud-q4-k-xl",
            {"inkling-2-mini-gguf-ud-q4-k-xl": 17629760736,
             "qwen2.5-7b-instruct-q4-k-m": 8500000000},
            {"id": "qwen2.5-7b-instruct-q4-k-m", "ready": True, "pid": 42, "uptime": "4m"},
            None)
        self.assertIn("* inkling-2-mini", text)
        self.assertIn("RUNNING", text)
        self.assertIn("PID 42", text)
        self.assertIn("selected model is not the running model", text)

    def test_missing_file_is_marked(self):
        text = render.format_list(PROFILES, "qwen2.5-7b-instruct-q4-k-m",
                                  {"qwen2.5-7b-instruct-q4-k-m": None,
                                   "inkling-2-mini-gguf-ud-q4-k-xl": 1}, None, None)
        self.assertIn("missing", text)

    def test_short_context(self):
        self.assertEqual(render.short_context(8192), "8K")
        self.assertEqual(render.short_context(65536), "64K")
        self.assertEqual(render.short_context(131072), "128K")
        self.assertEqual(render.short_context(8191), "8,191")

    def test_list_shows_context_column(self):
        text = render.format_list(
            PROFILES, "qwen2.5-7b-instruct-q4-k-m",
            {"qwen2.5-7b-instruct-q4-k-m": 1,
             "inkling-2-mini-gguf-ud-q4-k-xl": 1},
            None, None,
            contexts={"inkling-2-mini-gguf-ud-q4-k-xl": 65536})
        self.assertIn("64K", text)

    def test_status_shows_configured_and_server_context(self):
        running = {"id": "qwen2.5-7b-instruct-q4-k-m", "ready": True,
                   "pid": 42, "uptime": "4m"}
        text = render.format_status(
            "qwen2.5-7b-instruct-q4-k-m", running, None,
            "http://127.0.0.1:8080/v1", context=8192, server_context=8192)
        self.assertIn("8K", text)
        self.assertIn("server 8K", text)
        mismatch = render.format_status(
            "qwen2.5-7b-instruct-q4-k-m", running, None,
            "http://127.0.0.1:8080/v1", context=65536, server_context=8192)
        self.assertIn("warning", mismatch)

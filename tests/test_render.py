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

    def test_format_apply_changed_context(self):
        rows = [{"model": "qwen", "size": 17629760736, "old": 8192,
                 "new": 131072, "required": 30000000000, "action": "updated",
                 "approximate": False, "native": None}]
        text = render.format_apply(rows)
        self.assertIn("MODEL", text)
        self.assertIn("FILE", text)
        self.assertIn("CONTEXT", text)
        self.assertIn("EST. NEEDED", text)
        self.assertIn("ACTION", text)
        self.assertIn("qwen", text)
        self.assertIn("16.4 GiB", text)
        self.assertIn("8K \u2192 128K", text)
        self.assertIn("updated", text)

    def test_format_apply_unchanged_and_missing(self):
        rows = [
            {"model": "a", "size": 10, "old": 8192, "new": 8192,
             "required": None, "action": "kept", "approximate": False,
             "native": None},
            {"model": "gone", "size": None, "old": 8192, "new": 8192,
             "required": None, "action": "missing", "approximate": False,
             "native": None},
        ]
        text = render.format_apply(rows)
        self.assertIn("missing", text)
        self.assertNotIn("\u2192", text)

    def test_format_apply_approx_and_beyond_native(self):
        rows = [{"model": "m", "size": 10, "old": 8192, "new": 131072,
                 "required": 100, "action": "updated", "approximate": True,
                 "native": 32768}]
        text = render.format_apply(rows)
        self.assertIn("approx", text)
        self.assertIn("beyond native 32K", text)

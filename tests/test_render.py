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

    def test_runtime_summary(self):
        runtime = {
            "kv_type_k": "q8_0", "kv_type_v": "q8_0", "flash_attn": "on",
            "batch": 2048, "ubatch": 512, "parallel": 1, "cache_prompt": True,
            "cache_reuse": 256, "fit": True, "fit_target": 2048,
            "kv_unified": True, "no_kv_offload": False,
        }
        self.assertEqual(
            render.runtime_summary(runtime),
            "q8_0 KV \u00b7 2048/512 \u00b7 fit 2048 \u00b7 cache-reuse 256")

    def test_runtime_summary_asymmetric_and_off(self):
        runtime = {
            "kv_type_k": "q5_0", "kv_type_v": "q4_0",
            "batch": 1024, "ubatch": 256, "fit": False, "cache_reuse": 0,
        }
        text = render.runtime_summary(runtime)
        self.assertIn("q5_0/q4_0 KV", text)
        self.assertIn("1024/256", text)
        self.assertIn("no-fit", text)
        self.assertIn("cache-reuse 0", text)

    def test_format_apply_includes_kv_and_batch(self):
        rows = [{"model": "qwen", "size": 17629760736, "old": 8192,
                 "new": 131072, "required": 30000000000, "action": "updated",
                 "approximate": False, "native": None,
                 "kv": "q8_0", "batch": 2048, "ubatch": 512}]
        text = render.format_apply(rows)
        self.assertIn("KV", text)
        self.assertIn("BATCH", text)
        self.assertIn("q8_0", text)
        self.assertIn("2048/512", text)

    def test_format_apply_missing_has_empty_kv_and_batch(self):
        rows = [{"model": "gone", "size": None, "old": 8192, "new": 8192,
                 "required": None, "action": "missing", "approximate": False,
                 "native": None}]
        text = render.format_apply(rows)
        self.assertIn("KV", text)
        self.assertIn("BATCH", text)
        self.assertIn("missing", text)

    def test_format_analysis_columns(self):
        rows = [{"model": "qwen", "alias": "qwen", "size": 3_758_096_384,
                 "context": 65536, "f16": 3_758_096_384,
                 "q8_0": 1_996_488_704, "q4_0": 1_056_964_608,
                 "saving": 0.46875, "fits": True, "native": 32768}]
        text = render.format_analysis(rows, 65536)
        for column in ("MODEL", "SIZE", "CONTEXT", "f16", "q8_0", "q4_0",
                       "SAVE", "FIT"):
            self.assertIn(column, text)
        self.assertIn("46.9%", text)
        self.assertIn("yes", text)
        self.assertIn("beyond native 32K", text)

    def test_format_analysis_kv_cells_in_gib(self):
        rows = [{"model": "qwen", "alias": "qwen", "size": 3_758_096_384,
                 "context": 65536, "f16": 3_758_096_384,
                 "q8_0": 1_996_488_704, "q4_0": 1_056_964_608,
                 "saving": 0.46875, "fits": True, "native": 32768}]
        text = render.format_analysis(rows, 65536)
        self.assertIn("0.98 GiB", text)   # sub-GiB q4_0 still rendered in GiB
        self.assertIn("3.50 GiB", text)   # whole-GiB f16 keeps the GiB suffix
        self.assertNotIn("MiB", text)

    def test_format_analysis_renders_dashes(self):
        rows = [{"model": "gone", "alias": "gone", "size": None,
                 "context": 65536, "f16": None, "q8_0": None, "q4_0": None,
                 "saving": None, "fits": None, "native": None}]
        text = render.format_analysis(rows, 65536)
        self.assertIn("-", text)

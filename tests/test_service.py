import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Set before importing llm: `discover -s tests` imports modules top-level and
# does not run tests/__init__.py.
os.environ.setdefault("LMM_NO_CACHE", "1")

from llm import service

class LogTest(unittest.TestCase):
    def test_tail_reads_only_the_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("old\n" + "x" * 1000 + "\nlast line\n")
            text = service.tail_text(path, max_bytes=64)
            self.assertTrue(text.endswith("last line\n"))
            self.assertLess(len(text), 100)

    def test_rotate_at_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("a" * 200)
            self.assertTrue(service.rotate_log(path, max_bytes=100))
            self.assertFalse(path.exists())
            self.assertEqual((Path(tmp) / "server.log.1").stat().st_size, 200)
            self.assertFalse(service.rotate_log(Path(tmp) / "server.log.1", max_bytes=100))

    def test_last_failure_uses_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("E llama_model_load: error loading model: "
                            "unknown model architecture: 'inkling2_loop'\n")
            self.assertIn("1.1.0", service.last_failure(path))

class ArgsTest(unittest.TestCase):
    def test_server_args_carry_profile_values(self):
        profile = {"id": "x", "alias": "x-alias", "path": "/tmp/x.gguf", "context": 4096,
                   "sampling": {"temperature": 0.7, "top_p": 1.0, "min_p": 0.01,
                                "repeat_penalty": 1.0}}
        args = service.server_args(profile)
        self.assertIn("/tmp/x.gguf", args)
        self.assertIn("4096", args)
        self.assertIn("x-alias", args)
        self.assertIn("--port", args)

    def test_server_args_uses_config_llama_server(self):
        profile = {"id": "x", "alias": "x", "path": "/tmp/x.gguf", "context": 1,
                   "sampling": {"temperature": 0, "top_p": 1, "min_p": 0,
                                "repeat_penalty": 1}}
        with mock.patch.object(service.config, "LLAMA_SERVER", "/custom/llama-server"):
            self.assertEqual(service.server_args(profile)[0], "/custom/llama-server")

    def test_identity_uses_config_ps(self):
        with mock.patch.object(service.config, "PS", "/custom/ps"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "identity"
            service.identity(123)
        self.assertEqual(run.call_args[0][0][0], "/custom/ps")

    def test_listener_uses_config_lsof(self):
        with mock.patch.object(service.config, "LSOF", "/custom/lsof"), \
             mock.patch("subprocess.run") as run:
            run.return_value.stdout = "42\n"
            self.assertEqual(service.listener_pid(), 42)
        self.assertEqual(run.call_args[0][0][0], "/custom/lsof")

def _pairs(args):
    """Map each flag token to its value, or True for value-less flags."""
    result = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith("-") and index + 1 < len(args) \
                and not args[index + 1].startswith("-"):
            result[token] = args[index + 1]
            index += 2
            continue
        if token.startswith("-"):
            result[token] = True
        index += 1
    return result


class RuntimeArgsTest(unittest.TestCase):
    def _profile(self, runtime=..., **extra):
        profile = {"id": "x", "alias": "x-alias", "path": "/tmp/x.gguf",
                   "context": 4096,
                   "sampling": {"temperature": 0.7, "top_p": 1.0,
                                "min_p": 0.01, "repeat_penalty": 1.0}}
        if runtime is not ...:
            profile["runtime"] = runtime
        profile.update(extra)
        return profile

    def test_parse_help_collects_short_and_long_flags(self):
        text = ("-ctk,  --cache-type-k TYPE   KV cache data type for K\n"
                "-ctv,  --cache-type-v TYPE   KV cache data type for V\n"
                "-kvu,  --kv-unified, -no-kvu, --no-kv-unified\n"
                "-fa,   --flash-attn [on|off|auto]\n"
                "-fitt, --fit-target MiB0,MiB1,MiB2,...\n")
        flags = service.parse_help(text)
        self.assertIn("-ctk", flags)
        self.assertIn("--cache-type-k", flags)
        self.assertIn("-ctv", flags)
        self.assertIn("--cache-type-v", flags)
        self.assertIn("-kvu", flags)
        self.assertIn("--kv-unified", flags)
        self.assertIn("-fa", flags)
        self.assertIn("--flash-attn", flags)
        self.assertIn("--fit-target", flags)
        self.assertNotIn("TYPE", flags)

    def test_balanced_runtime_emits_kv_batch_fit_flags(self):
        profile = self._profile(runtime={
            "kv_type_k": "q8_0", "kv_type_v": "q8_0", "flash_attn": "on",
            "batch": 2048, "ubatch": 512, "parallel": 1,
            "cache_prompt": True, "cache_reuse": 256, "fit": True,
            "fit_target": 2048, "kv_unified": True, "no_kv_offload": False})
        supported = {"-ctk", "-ctv", "-kvu", "--cache-prompt",
                     "--no-cache-prompt", "--cache-reuse", "--fit",
                     "--fit-target", "--no-kv-offload"}
        pairs = _pairs(service.server_args(profile, supported=supported))
        self.assertEqual(pairs["-ctk"], "q8_0")
        self.assertEqual(pairs["-ctv"], "q8_0")
        self.assertEqual(pairs["-fa"], "on")
        self.assertEqual(pairs["-c"], "4096")
        self.assertEqual(pairs["-b"], "2048")
        self.assertEqual(pairs["-ub"], "512")
        self.assertEqual(pairs["--parallel"], "1")
        self.assertIs(pairs["-kvu"], True)
        self.assertIs(pairs["--cache-prompt"], True)
        self.assertEqual(pairs["--cache-reuse"], "256")
        self.assertEqual(pairs["--fit"], "on")
        self.assertEqual(pairs["--fit-target"], "2048")
        self.assertNotIn("--no-kv-offload", pairs)

    def test_quantized_kv_forces_flash_attention_on(self):
        profile = self._profile(runtime={
            "kv_type_k": "q4_0", "kv_type_v": "q4_0", "flash_attn": "off"})
        pairs = _pairs(service.server_args(
            profile, supported={"-ctk", "-ctv"}))
        self.assertEqual(pairs["-fa"], "on")
        self.assertEqual(pairs["-ctk"], "q4_0")
        self.assertEqual(pairs["-ctv"], "q4_0")

    def test_unsupported_flags_are_omitted(self):
        profile = self._profile(runtime={
            "kv_type_k": "q8_0", "kv_type_v": "q8_0", "batch": 2048,
            "ubatch": 512, "cache_prompt": True, "cache_reuse": 256,
            "fit": True, "fit_target": 2048, "kv_unified": True,
            "no_kv_offload": True})
        args = service.server_args(profile, supported=set())
        self.assertNotIn("-ctk", args)
        self.assertNotIn("-ctv", args)
        self.assertNotIn("-kvu", args)
        self.assertNotIn("--cache-prompt", args)
        self.assertNotIn("--cache-reuse", args)
        self.assertNotIn("--fit", args)
        self.assertNotIn("--no-kv-offload", args)
        pairs = _pairs(args)
        self.assertEqual(pairs["-fa"], "on")
        self.assertEqual(pairs["-b"], "2048")
        self.assertEqual(pairs["-ub"], "512")

    def test_legacy_profile_keeps_old_args(self):
        profile = self._profile()
        args = service.server_args(profile, supported={
            "-ctk", "-ctv", "-kvu", "--cache-prompt", "--no-cache-prompt",
            "--cache-reuse", "--fit", "--fit-target", "--no-kv-offload"})
        self.assertNotIn("-ctk", args)
        self.assertNotIn("-ctv", args)
        self.assertNotIn("-kvu", args)
        self.assertNotIn("--cache-prompt", args)
        self.assertNotIn("--cache-reuse", args)
        self.assertNotIn("--fit", args)
        self.assertNotIn("--no-kv-offload", args)
        pairs = _pairs(args)
        self.assertEqual(pairs["-fa"], "on")
        self.assertEqual(pairs["-b"], "512")
        self.assertEqual(pairs["-ub"], "256")
        self.assertEqual(pairs["--parallel"], "1")

    def test_no_kv_offload_and_no_cache_reuse(self):
        profile = self._profile(runtime={"no_kv_offload": True,
                                         "cache_reuse": 0})
        args = service.server_args(profile, supported={
            "--no-kv-offload", "--cache-reuse", "--cache-prompt",
            "--no-cache-prompt"})
        self.assertIn("--no-kv-offload", args)
        self.assertNotIn("--cache-reuse", args)
        self.assertIn("--cache-prompt", args)


class SafetyTest(unittest.TestCase):
    def test_stale_pid_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "process.json"
            pidfile.write_text(json.dumps({"pid": os.getpid(), "identity": "old-identity"}))
            with mock.patch.object(service, "PIDFILE", pidfile), \
                 mock.patch.object(service, "identity", return_value="new-identity"):
                self.assertIsNone(service.running())

    def test_foreign_listener_is_not_expected(self):
        profile = {"id": "x", "alias": "x-alias", "path": "/tmp/x.gguf"}
        with mock.patch.object(service, "selected_profile", return_value=profile), \
             mock.patch.object(service, "identity", return_value="/usr/bin/other-server"):
            self.assertFalse(service.is_expected_listener(999))

    def test_empty_model_file_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "model.gguf"
            empty.touch()
            with self.assertRaises(service.ModelLoadError):
                service.validate_profile_files({"id": "x", "path": str(empty)})


class PropsTest(unittest.TestCase):
    def test_server_context_reads_props(self):
        payload = json.dumps({"default_generation_settings": {"n_ctx": 8192}})
        response = io.StringIO(payload)
        fake_open = mock.MagicMock()
        fake_open.return_value.__enter__.return_value = response
        with mock.patch.object(service.HTTP, "open", fake_open):
            self.assertEqual(service.server_context(), 8192)

    def test_server_context_returns_none_when_unreachable(self):
        with mock.patch.object(service.HTTP, "open", side_effect=OSError):
            self.assertIsNone(service.server_context())

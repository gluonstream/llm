import unittest
from unittest import mock

from llm import host


class HostTest(unittest.TestCase):
    def test_find_tool_prefers_which(self):
        self.assertEqual(
            host.find_tool("lsof", "/fallback", which=lambda name: "/usr/bin/lsof"),
            "/usr/bin/lsof")

    def test_find_tool_uses_fallback(self):
        self.assertEqual(
            host.find_tool("lsof", "/fallback", which=lambda name: None),
            "/fallback")

    def test_llama_server_env_wins(self):
        self.assertEqual(
            host.find_llama_server(
                env={"LMM_LLAMA_SERVER": "/custom/llama-server"},
                which=lambda name: "/usr/bin/llama-server"),
            "/custom/llama-server")

    def test_llama_server_path_wins(self):
        self.assertEqual(
            host.find_llama_server(env={}, which=lambda name: "/usr/bin/llama-server"),
            "/usr/bin/llama-server")

    def test_llama_server_macos_and_arch_fallbacks(self):
        with mock.patch.object(host, "IS_MACOS", True):
            self.assertEqual(
                host.find_llama_server(env={}, which=lambda name: None),
                "/opt/homebrew/bin/llama-server")
        with mock.patch.object(host, "IS_MACOS", False):
            self.assertEqual(
                host.find_llama_server(env={}, which=lambda name: None),
                "/usr/bin/llama-server")

    def test_upgrade_hint_is_os_specific(self):
        with mock.patch.object(host, "IS_MACOS", True):
            self.assertIn("brew upgrade llama.cpp", host.llama_cpp_upgrade_hint())
        with mock.patch.object(host, "IS_MACOS", False):
            self.assertIn("Arch", host.llama_cpp_upgrade_hint())

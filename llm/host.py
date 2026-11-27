"""Operating-system and external-tool discovery for llm."""
import os
import shutil
import sys

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def find_tool(name, fallback, which=None):
    which = shutil.which if which is None else which
    return which(name) or fallback


def find_llama_server(env=None, which=None):
    env = os.environ if env is None else env
    configured = env.get("LMM_LLAMA_SERVER")
    if configured:
        return configured
    fallback = ("/opt/homebrew/bin/llama-server" if IS_MACOS
                else "/usr/bin/llama-server")
    return find_tool("llama-server", fallback, which=which)


def llama_cpp_upgrade_hint():
    if IS_MACOS:
        return "brew upgrade llama.cpp"
    return "update llama.cpp (Arch/AUR package or source build)"

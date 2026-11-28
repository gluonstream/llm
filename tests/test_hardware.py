import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import gguf, hardware


def model_metadata(**overrides):
    """Metadata whose KV cache is exactly 1 GiB at an 8192 context."""
    metadata = {
        "general.architecture": "test",
        "test.block_count": 32,
        "test.embedding_length": 2048,
        "test.attention.head_count": 32,
        "test.attention.head_count_kv": 16,
        "test.context_length": 32768,
    }
    metadata.update(overrides)
    return metadata


def machine(total_ram, unified=True, vram=None):
    return hardware.Hardware(chip="test", model="test", total_ram=total_ram,
                             unified=unified, gpu="test", vram=vram,
                             disk_free=0)


class HardwareTest(unittest.TestCase):
    def test_budget_and_estimate(self):
        self.assertEqual(hardware.memory_budget(36 * hardware.GIB),
                         int(36 * hardware.GIB * 0.8))
        self.assertEqual(hardware.estimate_required(16 * hardware.GIB, 8192),
                         16 * hardware.GIB + 2 * hardware.GIB + int(1.5 * hardware.GIB))

    def test_preflight_fits_and_refuses(self):
        machine = hardware.Hardware(chip="Apple M3 Max", model="Mac15,8",
                                    total_ram=36 * hardware.GIB, unified=True,
                                    gpu="Apple M3 Max", vram=None,
                                    disk_free=100 * hardware.GIB)
        self.assertEqual(hardware.preflight(16 * hardware.GIB, 8192,
                                            hardware=machine), [])
        warnings = hardware.preflight(30 * hardware.GIB, 8192, hardware=machine)
        self.assertTrue(any("does not fit" in warning for warning in warnings))

    def test_preflight_disk_warning(self):
        machine = hardware.Hardware(chip="x", model="x", total_ram=36 * hardware.GIB,
                                    unified=True, gpu="x", vram=None,
                                    disk_free=1 * hardware.GIB)
        warnings = hardware.preflight(16 * hardware.GIB, 8192, hardware=machine)
        self.assertTrue(any("disk" in warning for warning in warnings))

    def test_detect_hardware_with_injected_sysctl(self):
        values = {"hw.memsize": str(36 * hardware.GIB),
                  "machdep.cpu.brand_string": "Apple M3 Max",
                  "hw.model": "Mac15,8",
                  "hw.optional.arm64": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                sysctl=lambda name: values.get(name, ""),
                displays=lambda: ("Apple M3 Max", None),
                models_dir=Path(tmp),
                platform_name="darwin")
        self.assertEqual(machine.total_ram, 36 * hardware.GIB)
        self.assertTrue(machine.unified)
        self.assertEqual(machine.chip, "Apple M3 Max")

    def test_detect_falls_back_to_sysconf_without_sysctl(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("llm.hardware.os.sysconf",
                            side_effect=[4096, 1000]):
                machine = hardware.detect_hardware(
                    sysctl=lambda name: "",
                    displays=lambda: (None, None),
                    models_dir=Path(tmp),
                    platform_name="darwin")
        self.assertEqual(machine.total_ram, 4096 * 1000)

    def test_memory_budget_uses_available_ram(self):
        self.assertEqual(
            hardware.memory_budget(36 * hardware.GIB, unified=True,
                                   available=24 * hardware.GIB),
            int(24 * hardware.GIB * 0.8))

    def test_parse_meminfo(self):
        values = hardware._parse_meminfo(
            "MemTotal:       1000 kB\nMemAvailable:    500 kB\n")
        self.assertEqual(values["MemTotal"], 1000 * 1024)
        self.assertEqual(values["MemAvailable"], 500 * 1024)

    def test_linux_gpu_prefers_nvidia(self):
        def runner(command, timeout=5):
            if command[0] == "nvidia-smi":
                return "NVIDIA RTX 4090, 24564, 24000\n"
            return ""
        self.assertEqual(
            hardware._linux_gpu(runner=runner),
            ("NVIDIA RTX 4090", 24564 * 1024 * 1024))

    def test_linux_gpu_reads_amd_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = Path(tmp) / "card0" / "device"
            device.mkdir(parents=True)
            (device / "mem_info_vram_total").write_text("8589934592")
            def runner(command, timeout=5):
                return ""
            self.assertEqual(
                hardware._linux_gpu(sysfs_root=Path(tmp), runner=runner),
                ("AMD GPU", 8589934592))

    def test_linux_gpu_falls_back_to_lspci(self):
        with tempfile.TemporaryDirectory() as tmp:
            def runner(command, timeout=5):
                if command[0] == "lspci":
                    return ("00:02.0 VGA compatible controller: "
                            "Intel Corporation Xe\n")
                return ""
            name, vram = hardware._linux_gpu(sysfs_root=Path(tmp), runner=runner)
            self.assertIn("Intel", name)
            self.assertIsNone(vram)

    def test_linux_gpu_returns_none_when_all_probes_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                hardware._linux_gpu(sysfs_root=Path(tmp),
                                    runner=lambda command, timeout=5: ""),
                (None, None))

    def test_detect_hardware_linux_uses_injected_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="linux",
                linux_cpu=lambda: "AMD Ryzen 9",
                linux_ram=lambda: (32 * hardware.GIB, 24 * hardware.GIB),
                linux_gpu=lambda: ("NVIDIA RTX 4090", 24 * hardware.GIB),
                models_dir=Path(tmp))
        self.assertEqual(machine.total_ram, 32 * hardware.GIB)
        self.assertEqual(machine.available_ram, 24 * hardware.GIB)
        self.assertEqual(machine.chip, "AMD Ryzen 9")
        self.assertFalse(machine.unified)

    def test_detect_hardware_macos_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="darwin",
                sysctl=lambda name: {"hw.memsize": str(36 * hardware.GIB),
                                     "hw.model": "Mac15,8",
                                     "hw.optional.arm64": "1"}.get(name, ""),
                displays=lambda: ("Apple M3 Max", None),
                models_dir=Path(tmp))
        self.assertTrue(machine.unified)
        self.assertEqual(machine.available_ram, 0)

    def test_detect_hardware_linux_with_empty_probes_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="linux",
                linux_cpu=lambda: "",
                linux_ram=lambda: (0, 0),
                linux_gpu=lambda: (None, None),
                models_dir=Path(tmp))
        self.assertTrue(machine.chip)
        self.assertEqual(machine.total_ram, 0)
        self.assertFalse(machine.unified)

    def test_disk_free_uses_btrfs_when_available(self):
        with mock.patch.object(hardware.host, "IS_LINUX", True), \
             mock.patch.object(hardware, "_filesystem_type", return_value="btrfs"), \
             mock.patch.object(hardware, "_btrfs_free", return_value=123), \
             mock.patch.object(hardware.shutil, "disk_usage") as usage:
            usage.return_value.free = 999
            self.assertEqual(hardware._disk_free("/"), 123)

    def test_context_budget_unified(self):
        machine_under_test = machine(36 * hardware.GIB, unified=True, vram=None)
        self.assertEqual(hardware.context_budget(machine_under_test),
                         int(36 * hardware.GIB * 0.80))

    def test_context_budget_discrete_uses_total_vram(self):
        machine_under_test = machine(64 * hardware.GIB, unified=False,
                                     vram=24 * hardware.GIB)
        self.assertEqual(hardware.context_budget(machine_under_test),
                         int(24 * hardware.GIB * 0.90))

    def test_estimate_required_with_metadata_uses_kv(self):
        metadata = model_metadata()
        self.assertEqual(gguf.kv_cache_bytes(metadata, 8192), hardware.GIB)
        self.assertEqual(
            hardware.estimate_required(4 * hardware.GIB, 8192, metadata),
            4 * hardware.GIB + hardware.GIB + hardware.GIB + 256 * 1024 * 1024)

    def test_estimate_required_typed_kv(self):
        metadata = model_metadata()
        f16 = hardware.estimate_required(4 * hardware.GIB, 8192, metadata)
        q8 = hardware.estimate_required(4 * hardware.GIB, 8192, metadata,
                                        "q8_0", "q8_0")
        # The q8_0 KV cache is 1.0625/2.0 of the f16 cache, so the estimate
        # drops by exactly the saved KV bytes.
        saved = hardware.GIB - int(hardware.GIB * 1.0625 / 2.0)
        self.assertEqual(q8, f16 - saved)

    def test_choose_context_prefers_128k_when_it_fits(self):
        choice = hardware.choose_context(4 * hardware.GIB, 8192,
                                         machine(36 * hardware.GIB),
                                         model_metadata())
        self.assertEqual(choice.context, 131072)
        self.assertEqual(choice.reason, "fits")
        self.assertEqual(choice.budget, int(36 * hardware.GIB * 0.80))
        self.assertEqual(choice.required,
                         4 * hardware.GIB + 16 * hardware.GIB + hardware.GIB
                         + 256 * 1024 * 1024)
        self.assertFalse(choice.approximate)

    def test_choose_context_falls_back_to_64k(self):
        choice = hardware.choose_context(4 * hardware.GIB, 8192,
                                         machine(20 * hardware.GIB),
                                         model_metadata())
        self.assertEqual(choice.context, 65536)
        self.assertEqual(choice.reason, "fits")

    def test_choose_context_keeps_current_when_none_fit(self):
        choice = hardware.choose_context(4 * hardware.GIB, 32768,
                                         machine(8 * hardware.GIB),
                                         model_metadata())
        self.assertEqual(choice.context, 32768)
        self.assertIn("no candidate fits", choice.reason)

    def test_choose_context_ceiling_caps_candidates(self):
        choice = hardware.choose_context(4 * hardware.GIB, 8192,
                                         machine(36 * hardware.GIB),
                                         model_metadata(), ceiling=65536)
        self.assertEqual(choice.context, 65536)
        self.assertNotEqual(choice.context, 131072)

    def test_choose_context_marks_approximate_without_metadata(self):
        choice = hardware.choose_context(4 * hardware.GIB, 8192,
                                         machine(36 * hardware.GIB), None)
        self.assertTrue(choice.approximate)
        self.assertIsNone(choice.native)
        self.assertIn("approximate", choice.reason)

    def test_choose_context_reports_native_window(self):
        choice = hardware.choose_context(4 * hardware.GIB, 8192,
                                         machine(36 * hardware.GIB),
                                         model_metadata())
        self.assertEqual(choice.native, 32768)

    def test_parse_gpustat_reads_usage_and_processes(self):
        payload = json.dumps({"gpus": [
            {"index": 0,
             "name": "NVIDIA RTX 4090",
             "memory.total": 24564,
             "memory.used": 3200,
             "memory.unified": False,
             "utilization.gpu": 35,
             "processes": [{"pid": 4242, "command": "llama-server",
                            "gpu_memory_usage": 1234, "username": "davide"}]},
            {"index": 1,
             "name": "NVIDIA RTX 3090",
             "memory.total": 24576,
             "memory.used": None,
             "utilization.gpu": None,
             "processes": []},
        ]})
        gpus = hardware.parse_gpustat(payload)
        self.assertEqual(len(gpus), 2)
        first = gpus[0]
        self.assertEqual(first.name, "NVIDIA RTX 4090")
        self.assertEqual(first.total, 24564 * 1024 * 1024)
        self.assertEqual(first.used, 3200 * 1024 * 1024)
        self.assertEqual(first.free, (24564 - 3200) * 1024 * 1024)
        self.assertEqual(first.utilization, 35)
        self.assertEqual(first.processes[0]["pid"], 4242)
        self.assertFalse(first.unified)
        self.assertEqual(first.source, "gpustat")
        self.assertIsNone(gpus[1].free)
        self.assertIsNone(gpus[1].used)
        self.assertIsNone(gpus[1].utilization)

    def test_parse_gpustat_rejects_bad_json(self):
        self.assertIsNone(hardware.parse_gpustat("not json"))
        self.assertIsNone(hardware.parse_gpustat("{}"))

    def test_parse_gpustat_reads_unified_flag(self):
        payload = json.dumps({"gpus": [{
            "name": "NVIDIA GB10",
            "memory.total": 131072,
            "memory.used": 8192,
            "memory.unified": True,
            "utilization.gpu": 5,
            "processes": [],
        }]})
        gpus = hardware.parse_gpustat(payload)
        self.assertEqual(gpus[0].index, 0)
        self.assertTrue(gpus[0].unified)
        self.assertEqual(gpus[0].free, (131072 - 8192) * 1024 * 1024)

    def test_parse_gpustat_detects_unified_by_name_without_flag(self):
        payload = json.dumps({"gpus": [
            {"name": "NVIDIA GB10",
             "memory.total": 131072,
             "memory.used": 8192,
             "processes": []},
            {"name": "NVIDIA RTX 4090",
             "memory.total": 24564,
             "memory.used": 3200,
             "processes": []},
        ]})
        gpus = hardware.parse_gpustat(payload)
        self.assertTrue(gpus[0].unified)
        self.assertFalse(gpus[1].unified)

    def test_device_tree_marks_tegra_unified(self):
        with tempfile.TemporaryDirectory() as tmp:
            compatible = Path(tmp) / "compatible"
            compatible.write_text("nvidia,tegra234\0nvidia,tegra\0")
            self.assertTrue(hardware._device_tree_unified(compatible))
            other = Path(tmp) / "other"
            other.write_text("apple,arm-platform\0")
            self.assertFalse(hardware._device_tree_unified(other))

    def test_detect_gpus_returns_empty_when_all_probes_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                hardware.detect_gpus(runner=lambda command, timeout=5: "",
                                     sysfs_root=Path(tmp),
                                     displays=lambda: (None, None)),
                [])

    def test_gpustat_falls_back_to_nvidia_smi(self):
        def runner(command, timeout=5):
            if command[0] == "nvidia-smi":
                return "NVIDIA RTX 4090, 24564, 3200, 21364, 35\n"
            return ""

        with tempfile.TemporaryDirectory() as tmp:
            gpus = hardware.detect_gpus(runner=runner, sysfs_root=Path(tmp))
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].name, "NVIDIA RTX 4090")
        self.assertEqual(gpus[0].source, "nvidia-smi")

    def test_nvidia_smi_reads_used_free_util(self):
        gpus = hardware.nvidia_smi_gpus(
            runner=lambda command, timeout=5:
            "NVIDIA RTX 4090, 24564, 3200, 21364, 35\n")
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        self.assertEqual(gpu.total, 24564 * 1024 * 1024)
        self.assertEqual(gpu.used, 3200 * 1024 * 1024)
        self.assertEqual(gpu.free, 21364 * 1024 * 1024)
        self.assertEqual(gpu.utilization, 35)
        self.assertEqual(gpu.source, "nvidia-smi")

    def test_amd_sysfs_reads_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = Path(tmp) / "card0" / "device"
            device.mkdir(parents=True)
            (device / "mem_info_vram_total").write_text("8589934592")
            (device / "mem_info_vram_used").write_text("1073741824")
            gpus = hardware.amd_sysfs_gpus(sysfs_root=Path(tmp))
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        self.assertEqual(gpu.name, "AMD GPU")
        self.assertEqual(gpu.total, 8589934592)
        self.assertEqual(gpu.used, 1073741824)
        self.assertEqual(gpu.free, 8589934592 - 1073741824)
        self.assertEqual(gpu.source, "sysfs")

    def test_unified_gpu_name(self):
        self.assertTrue(hardware.is_unified_gpu_name("NVIDIA GB10"))
        self.assertTrue(hardware.is_unified_gpu_name("Jetson Orin"))
        self.assertFalse(hardware.is_unified_gpu_name("NVIDIA RTX 4090"))

    def test_detect_hardware_linux_marks_unified_gb10(self):
        stats = hardware.GpuStats(index=0, name="NVIDIA GB10",
                                  total=128 * hardware.GIB,
                                  used=8 * hardware.GIB,
                                  free=120 * hardware.GIB,
                                  utilization=5, unified=True,
                                  source="gpustat")
        with tempfile.TemporaryDirectory() as tmp:
            machine = hardware.detect_hardware(
                platform_name="linux",
                linux_cpu=lambda: "ARM",
                linux_ram=lambda: (128 * hardware.GIB, 120 * hardware.GIB),
                linux_gpu=lambda: ("NVIDIA GB10", 128 * hardware.GIB),
                gpu_stats=lambda: [stats],
                models_dir=Path(tmp))
        self.assertTrue(machine.unified)
        self.assertEqual(machine.gpu, "NVIDIA GB10")
        self.assertEqual(machine.vram, 128 * hardware.GIB)
        self.assertEqual(machine.gpus, [stats])

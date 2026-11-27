import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import hardware

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

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
                models_dir=Path(tmp))
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
                    models_dir=Path(tmp))
        self.assertEqual(machine.total_ram, 4096 * 1000)

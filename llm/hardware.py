"""Detect local hardware and estimate whether a model will fit."""
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import host, render

GIB = 1024 ** 3
DISK_HEADROOM = 512 * 1024 * 1024


@dataclass
class Hardware:
    chip: str
    model: str
    total_ram: int
    unified: bool
    gpu: str
    vram: int
    disk_free: int
    available_ram: int = 0


def _run(command, timeout=5):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or "").strip()


def _sysctl(name):
    return _run(["sysctl", "-n", name])


def _displays():
    text = _run(["system_profiler", "SPDisplaysDataType"], timeout=15)
    gpu = None
    vram = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:") and gpu is None:
            gpu = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("VRAM") and vram is None:
            match = re.search(r"(\d+)\s*(GB|MB)", stripped)
            if match:
                value = int(match.group(1))
                vram = value * (GIB if match.group(2) == "GB" else 1024 * 1024)
    return gpu, vram


def _existing_parent(path):
    current = Path(path or ".").expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        match = re.search(r"(\d+)\s*kB", rest)
        if match:
            values[key.strip()] = int(match.group(1)) * 1024
    return values


def _linux_ram():
    values = _parse_meminfo(_read_text("/proc/meminfo"))
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def _linux_cpu():
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def _linux_gpu(sysfs_root=None, runner=None):
    runner = runner or _run
    output = runner(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                     "--format=csv,noheader,nounits"])
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                return parts[0], int(parts[1]) * 1024 * 1024
            except ValueError:
                break
    largest = None
    for path in Path(sysfs_root or "/sys/class/drm").glob(
            "card*/device/mem_info_vram_total"):
        try:
            value = int(_read_text(path).strip())
        except ValueError:
            continue
        if largest is None or value > largest:
            largest = value
    if largest is not None:
        return "AMD GPU", largest
    for line in (runner(["lspci"]) or "").splitlines():
        if ("VGA compatible controller" in line
                or "3D controller" in line
                or "display controller" in line):
            _, separator, name = line.partition(": ")
            if separator:
                return name.strip(), None
    return None, None


def _filesystem_type(path, runner=None):
    runner = runner or _run
    output = (runner(["stat", "-f", "-c", "%T", str(path)]) or "").strip()
    return output or None


def _btrfs_free(path, runner=None):
    runner = runner or _run
    output = runner(["btrfs", "filesystem", "usage", "-b", str(path)])
    for line in (output or "").splitlines():
        match = re.search(r"Free \(estimated\):\s*(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _disk_free(path, runner=None):
    try:
        free = shutil.disk_usage(_existing_parent(path)).free
    except OSError:
        return None
    if host.IS_LINUX and _filesystem_type(path, runner) == "btrfs":
        btrfs_free = _btrfs_free(path, runner)
        if btrfs_free is not None:
            return btrfs_free
    return free


def detect_hardware(sysctl=None, displays=None, models_dir=None, include_gpu=True,
                    platform_name=None, linux_cpu=None, linux_ram=None,
                    linux_gpu=None):
    platform_name = platform_name or sys.platform
    target = models_dir or Path.home()
    if platform_name.startswith("linux"):
        cpu = linux_cpu or _linux_cpu
        ram = linux_ram or _linux_ram
        gpu_probe = linux_gpu or _linux_gpu
        try:
            total_ram, available_ram = ram()
        except (TypeError, ValueError):
            total_ram, available_ram = 0, 0
        chip = cpu() or platform.processor() or "unknown"
        if include_gpu:
            gpu, vram = gpu_probe()
        else:
            gpu, vram = None, None
        return Hardware(chip=chip, model=platform.machine() or "unknown",
                        total_ram=total_ram, unified=False,
                        gpu=gpu or "unknown", vram=vram,
                        disk_free=_disk_free(target),
                        available_ram=available_ram or 0)
    sysctl = sysctl or _sysctl
    displays = displays or _displays
    try:
        total_ram = int(sysctl("hw.memsize") or 0)
    except ValueError:
        total_ram = 0
    if total_ram <= 0:
        try:
            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total_ram = 0
    chip = (sysctl("machdep.cpu.brand_string")
            or sysctl("hw.model") or platform.processor() or "unknown")
    model = sysctl("hw.model") or "unknown"
    unified = sysctl("hw.optional.arm64").strip() == "1" or platform.machine() == "arm64"
    if include_gpu:
        gpu, vram = displays()
    else:
        gpu, vram = None, None
    if not gpu and unified:
        gpu = chip
    return Hardware(chip=chip, model=model, total_ram=total_ram, unified=unified,
                    gpu=gpu or "unknown", vram=vram, disk_free=_disk_free(target),
                    available_ram=0)


def memory_budget(total_ram, unified=True, available=None):
    if total_ram <= 0:
        return None
    base = available if available and available > 0 else total_ram
    return int(base * (0.80 if unified else 0.75))


def estimate_required(model_size, context=8192):
    overhead = 2 * GIB + int((max(context, 1) / 8192.0) * 1.5 * GIB)
    return model_size + overhead


def preflight(model_size, context=8192, models_dir=None, hardware=None):
    if not model_size:
        return []
    machine = hardware or detect_hardware(models_dir=models_dir, include_gpu=False)
    warnings = []
    required = estimate_required(model_size, context)
    budget = memory_budget(machine.total_ram, machine.unified,
                           machine.available_ram)
    if budget and required > budget:
        warnings.append(
            "model does not fit: needs ~%s (%s file + %s overhead at %d context), "
            "budget ~%s on %s (%s RAM)"
            % (render.human_size(required), render.human_size(model_size),
               render.human_size(required - model_size), context,
               render.human_size(budget), machine.chip,
               render.human_size(machine.total_ram)))
    if machine.disk_free is not None:
        needed = model_size + DISK_HEADROOM
        if machine.disk_free < needed:
            warnings.append(
                "not enough disk space: %s free, need %s"
                % (render.human_size(machine.disk_free), render.human_size(needed)))
    return warnings

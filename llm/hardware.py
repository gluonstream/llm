"""Detect local hardware and estimate whether a model will fit."""
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import render

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


def detect_hardware(sysctl=None, displays=None, models_dir=None, include_gpu=True):
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
    try:
        disk_free = shutil.disk_usage(_existing_parent(models_dir or Path.home())).free
    except OSError:
        disk_free = None
    return Hardware(chip=chip, model=model, total_ram=total_ram, unified=unified,
                    gpu=gpu or "unknown", vram=vram, disk_free=disk_free)


def memory_budget(total_ram, unified=True):
    if total_ram <= 0:
        return None
    return int(total_ram * (0.80 if unified else 0.75))


def estimate_required(model_size, context=8192):
    overhead = 2 * GIB + int((max(context, 1) / 8192.0) * 1.5 * GIB)
    return model_size + overhead


def preflight(model_size, context=8192, models_dir=None, hardware=None):
    if not model_size:
        return []
    machine = hardware or detect_hardware(models_dir=models_dir, include_gpu=False)
    warnings = []
    required = estimate_required(model_size, context)
    budget = memory_budget(machine.total_ram, machine.unified)
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

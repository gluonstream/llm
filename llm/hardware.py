"""Detect local hardware and estimate whether a model will fit."""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import gguf, host, render

GIB = 1024 ** 3
MIB = 1024 * 1024
DISK_HEADROOM = 512 * 1024 * 1024
DEVICE_TREE_COMPATIBLE = "/proc/device-tree/compatible"
UNIFIED_GPU_PATTERN = re.compile(r"\b(gb10|jetson|orin|xavier|tegra)\b",
                                 re.IGNORECASE)
UNIFIED_GPU_WORDS = ("gb10", "jetson", "orin", "xavier", "tegra")

# Context tiers. 12 GiB of usable VRAM/RAM is enough for 32k; larger
# machines get 64k, and very large unified/discrete memory gets 128k.
SMALL_CONTEXT = 32768
DEFAULT_CONTEXT = 65536
LARGE_CONTEXT = 131072
SMALL_MEMORY = 12 * GIB
LARGE_MEMORY = 128 * GIB


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
    gpus: list = field(default_factory=list)


@dataclass
class GpuStats:
    index: int = 0
    name: str = ""
    total: Optional[int] = None
    used: Optional[int] = None
    free: Optional[int] = None
    utilization: Optional[float] = None
    processes: list = field(default_factory=list)
    unified: bool = False
    source: str = "unknown"


@dataclass
class ContextChoice:
    context: int
    reason: str
    required: Optional[int]
    budget: Optional[int]
    approximate: bool
    native: Optional[int]


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


def _read_int(path):
    try:
        return int(_read_text(path).strip())
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _mib_to_bytes(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("GPU memory must be a number")
    return int(value) * MIB


def _gpu_number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("GPU metric must be a number")
    return value


def _sum_totals(gpus):
    totals = [gpu.total for gpu in gpus if gpu.total is not None]
    return sum(totals) if totals else None


def is_unified_gpu_name(name):
    if not isinstance(name, str) or not name:
        return False
    return bool(UNIFIED_GPU_PATTERN.search(name))


def _device_tree_unified(path=None):
    text = _read_text(path or DEVICE_TREE_COMPATIBLE).lower()
    return "nvidia" in text and any(word in text for word in UNIFIED_GPU_WORDS)


def _linux_gpu_stats(name, vram):
    if not name:
        return []
    if vram is None:
        source = "lspci"
    elif name == "AMD GPU":
        source = "sysfs"
    else:
        source = "nvidia-smi"
    return [GpuStats(index=0, name=name, total=vram,
                     unified=is_unified_gpu_name(name), source=source)]


def _display_gpus(name, vram):
    if not name:
        return []
    return [GpuStats(index=0, name=name, total=vram,
                     unified=is_unified_gpu_name(name),
                     source="system_profiler")]


def parse_gpustat(payload):
    try:
        data = json.loads(payload)
        if not isinstance(data, dict):
            return None
        entries = data.get("gpus")
        if not isinstance(entries, list):
            return None
        gpus = []
        for entry in entries:
            if not isinstance(entry, dict):
                return None
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                return None
            total = _mib_to_bytes(entry.get("memory.total"))
            used = _mib_to_bytes(entry.get("memory.used"))
            free = (max(total - used, 0)
                    if total is not None and used is not None else None)
            raw_processes = entry.get("processes") or []
            if not isinstance(raw_processes, list):
                return None
            processes = []
            for process in raw_processes:
                if not isinstance(process, dict):
                    return None
                processes.append({
                    "pid": process.get("pid"),
                    "command": process.get("command"),
                    "gpu_memory_usage": process.get("gpu_memory_usage"),
                })
            gpus.append(GpuStats(
                index=int(entry.get("index", 0)),
                name=name.strip(),
                total=total,
                used=used,
                free=free,
                utilization=_gpu_number(entry.get("utilization.gpu")),
                processes=processes,
                unified=(bool(entry.get("memory.unified", False))
                         or is_unified_gpu_name(name)),
                source="gpustat",
            ))
        return gpus
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def gpustat_gpus(runner=None):
    runner = runner or _run
    output = runner(["gpustat", "--json"])
    if not output or not output.strip():
        return None
    return parse_gpustat(output)


def nvidia_smi_gpus(runner=None):
    runner = runner or _run
    output = runner([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"])
    gpus = []
    for index, line in enumerate((output or "").splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        total_mib = _int_or_none(parts[1])
        used_mib = _int_or_none(parts[2])
        free_mib = _int_or_none(parts[3])
        total = total_mib * MIB if total_mib is not None else None
        used = used_mib * MIB if used_mib is not None else None
        free = free_mib * MIB if free_mib is not None else None
        if free is None and total is not None and used is not None:
            free = max(total - used, 0)
        utilization = _int_or_none(parts[4]) if len(parts) > 4 else None
        gpus.append(GpuStats(
            index=index,
            name=parts[0],
            total=total,
            used=used,
            free=free,
            utilization=utilization,
            unified=is_unified_gpu_name(parts[0]),
            source="nvidia-smi",
        ))
    return gpus or None


def amd_sysfs_gpus(sysfs_root=None):
    root = Path(sysfs_root) if sysfs_root else Path("/sys/class/drm")
    gpus = []
    for index, device in enumerate(sorted(root.glob("card*/device"))):
        total = _read_int(device / "mem_info_vram_total")
        if total is None:
            continue
        used = _read_int(device / "mem_info_vram_used")
        free = max(total - used, 0) if used is not None else None
        gpus.append(GpuStats(index=index, name="AMD GPU", total=total,
                             used=used, free=free, unified=False,
                             source="sysfs"))
    return gpus or None


def detect_gpus(runner=None, sysfs_root=None, displays=None):
    gpus = gpustat_gpus(runner)
    if gpus:
        return gpus
    gpus = nvidia_smi_gpus(runner)
    if gpus:
        return gpus
    gpus = amd_sysfs_gpus(sysfs_root)
    if gpus:
        return gpus
    if host.IS_LINUX:
        name, vram = _linux_gpu(sysfs_root=sysfs_root, runner=runner)
        gpus = _linux_gpu_stats(name, vram)
        if gpus:
            return gpus
    if host.IS_MACOS:
        gpu, vram = (displays or _displays)()
        gpus = _display_gpus(gpu, vram)
        if gpus:
            return gpus
    return []


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
                    linux_gpu=None, gpu_stats=None):
    platform_name = platform_name or sys.platform
    target = models_dir or Path.home()
    if platform_name.startswith("linux"):
        cpu = linux_cpu or _linux_cpu
        ram = linux_ram or _linux_ram
        try:
            total_ram, available_ram = ram()
        except (TypeError, ValueError):
            total_ram, available_ram = 0, 0
        chip = cpu() or platform.processor() or "unknown"
        if include_gpu:
            if gpu_stats:
                gpus = list(gpu_stats() or [])
            elif linux_gpu:
                name, vram = linux_gpu()
                gpus = _linux_gpu_stats(name, vram)
            else:
                gpus = detect_gpus()
            gpu = next((item.name for item in gpus if item.name), None)
            vram = _sum_totals(gpus)
            unified = any(item.unified for item in gpus)
            if not unified:
                unified = _device_tree_unified()
        else:
            gpu, vram, gpus, unified = None, None, [], False
        return Hardware(chip=chip, model=platform.machine() or "unknown",
                        total_ram=total_ram, unified=unified,
                        gpu=gpu or "unknown", vram=vram,
                        disk_free=_disk_free(target),
                        available_ram=available_ram or 0, gpus=gpus)
    display_probe = displays
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
        if gpu_stats:
            gpus = list(gpu_stats() or [])
        elif display_probe:
            name, vram = display_probe()
            gpus = _display_gpus(name, vram)
        else:
            gpus = detect_gpus(displays=displays)
        gpu = next((item.name for item in gpus if item.name), None)
        vram = _sum_totals(gpus)
    else:
        gpu, vram, gpus = None, None, []
    if not gpu and unified:
        gpu = chip
    return Hardware(chip=chip, model=model, total_ram=total_ram, unified=unified,
                    gpu=gpu or "unknown", vram=vram, disk_free=_disk_free(target),
                    available_ram=0, gpus=gpus)


def memory_budget(total_ram, unified=True, available=None):
    if total_ram <= 0:
        return None
    base = available if available and available > 0 else total_ram
    return int(base * (0.80 if unified else 0.75))


def context_budget(machine):
    """Return the capacity budget for model weights plus KV cache, or None."""
    if machine.unified:
        return memory_budget(machine.total_ram, True, machine.available_ram)
    if machine.vram:
        return int(machine.vram * 0.90)
    return memory_budget(machine.total_ram, False, machine.available_ram)


def estimate_required(model_size, context=8192, metadata=None,
                      kv_type_k="f16", kv_type_v="f16"):
    if metadata:
        kv_bytes = gguf.kv_cache_bytes(metadata, context, kv_type_k, kv_type_v)
        if kv_bytes is not None:
            overhead = max(GIB, int(model_size * 0.05)) + 256 * 1024 * 1024
            return model_size + kv_bytes + overhead
    overhead = 2 * GIB + int((max(context, 1) / 8192.0) * 1.5 * GIB)
    return model_size + overhead


def choose_context(model_size, current, machine, metadata=None,
                   candidates=(131072, 65536), ceiling=None,
                   kv_type_k="f16", kv_type_v="f16"):
    approximate = metadata is None
    native = gguf.native_context(metadata) if metadata else None
    budget = context_budget(machine)
    if budget is None:
        return ContextChoice(current, "memory budget unknown", None, None,
                             approximate, native)
    values = [current] + [value for value in candidates if value != current]
    values.sort(reverse=True)
    capped_current = ceiling is not None and current > ceiling
    if ceiling is not None:
        values = [value for value in values if value <= ceiling]
    for value in values:
        required = estimate_required(model_size, value, metadata,
                                     kv_type_k, kv_type_v)
        if required <= budget:
            reason = "fits (approximate)" if approximate else "fits"
            return ContextChoice(value, reason, required, budget, approximate,
                                 native)
    required = estimate_required(model_size, current, metadata,
                                 kv_type_k, kv_type_v)
    if capped_current:
        reason = "no candidate <= ceiling fits; keeping current"
    else:
        reason = "no candidate fits (needs ~%s, budget ~%s)" % (
            render.human_size(required), render.human_size(budget))
    return ContextChoice(current, reason, required, budget, approximate, native)


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

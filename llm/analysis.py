"""Compute the KV memory table for ``llm analyze``."""
from pathlib import Path

from . import gguf, hardware, hfcache


def _local_size(profile):
    """Return the on-disk size of a profile's model file(s), or None."""
    path = profile.get("path")
    if not path:
        return None
    model = Path(path)
    parts = (hfcache.split_set(model.name)
             or hfcache.split_set(profile.get("filename") or ""))
    try:
        if parts is not None:
            stem, _index, total = parts
            shards = model.parent.glob("%s-*-of-%05d.gguf" % (stem, total))
            size = sum(shard.stat().st_size for shard in shards)
            return size if size > 0 else None
        size = model.stat().st_size
        return size if size > 0 else None
    except OSError:
        return None


def rows(profiles, context=65536, machine=None):
    """Return one dict per profile with KV memory for three quantisations.

    Missing or unreadable metadata still yields a row, with the derived
    values left as ``None``. Order follows the input mapping.
    """
    result = []
    seen = set()
    for profile_id, profile in profiles.items():
        if profile_id in seen:
            continue
        seen.add(profile_id)
        size = _local_size(profile)
        metadata = (gguf.read_metadata(profile["path"])
                    if profile.get("path") else None)
        row = {
            "model": profile_id,
            "alias": profile.get("alias"),
            "size": size,
            "context": context,
            "f16": None,
            "q8_0": None,
            "q4_0": None,
            "saving": None,
            "fits": None,
            "native": None,
        }
        if metadata is not None:
            row["f16"] = gguf.kv_cache_bytes(metadata, context, "f16", "f16")
            row["q8_0"] = gguf.kv_cache_bytes(metadata, context,
                                              "q8_0", "q8_0")
            row["q4_0"] = gguf.kv_cache_bytes(metadata, context,
                                              "q4_0", "q4_0")
            row["native"] = gguf.native_context(metadata)
            if row["f16"] and row["q8_0"] is not None:
                row["saving"] = 1.0 - (row["q8_0"] / row["f16"])
        if size is not None and machine is not None and metadata is not None:
            budget = hardware.context_budget(machine)
            if budget is not None:
                required = hardware.estimate_required(
                    size, context, metadata, "q8_0", "q8_0")
                row["fits"] = required <= budget
        result.append(row)
    return result

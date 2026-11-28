"""Runtime tuning policies: pick context and server flags for llama.cpp."""
from dataclasses import dataclass
from typing import Optional

from . import hardware

# Shared fields for every policy.
_COMMON = {
    "flash_attn": "on",
    "parallel": 1,
    "cache_prompt": True,
    "cache_reuse": 256,
    "fit": True,
    "fit_target": 2048,
    "kv_unified": True,
    "no_kv_offload": False,
}


def _policy(kv_type_k, kv_type_v, candidates, batch, ubatch):
    policy = dict(_COMMON)
    policy.update({
        "kv_type_k": kv_type_k,
        "kv_type_v": kv_type_v,
        "candidates": list(candidates),
        "batch": batch,
        "ubatch": ubatch,
    })
    return policy


POLICIES = {
    "conservative": _policy("f16", "f16", [65536], 1024, 256),
    "balanced": _policy("q8_0", "q8_0", [131072, 65536], 2048, 512),
    "extended": _policy("q8_0", "q8_0", [131072], 2048, 1024),
    "aggressive": _policy("q4_0", "q4_0", [131072], 2048, 1024),
}

DEFAULT_POLICY = "balanced"

_RUNTIME_KEYS = (
    "kv_type_k", "kv_type_v", "flash_attn", "batch", "ubatch", "parallel",
    "cache_prompt", "cache_reuse", "fit", "fit_target", "kv_unified",
    "no_kv_offload",
)


@dataclass
class TunedChoice:
    context: int
    runtime: dict
    reason: str
    required: Optional[int]
    budget: Optional[int]
    approximate: bool
    native: Optional[int]


def policy_runtime(policy="balanced", overrides=None):
    """Return the runtime dict for a policy, with ``overrides`` applied."""
    base = POLICIES[policy]
    runtime = {key: base[key] for key in _RUNTIME_KEYS}
    for key, value in (overrides or {}).items():
        if key == "kv":
            runtime["kv_type_k"] = value
            runtime["kv_type_v"] = value
        elif key in ("kv_k",):
            runtime["kv_type_k"] = value
        elif key in ("kv_v",):
            runtime["kv_type_v"] = value
        elif key == "no_fit":
            runtime["fit"] = not value
        elif key == "no_cache_reuse":
            runtime["cache_reuse"] = 0 if value else runtime["cache_reuse"]
        elif key in _RUNTIME_KEYS:
            runtime[key] = value
    return runtime


def candidates_for(policy="balanced", observed=None):
    """Return context candidates, filtered to hold ``observed * 1.5``."""
    candidates = list(POLICIES[policy]["candidates"])
    if observed and observed > 0:
        threshold = int(observed * 1.5)
        filtered = [value for value in candidates if value >= threshold]
        if filtered:
            return filtered
    return candidates


def choose_runtime(model_size, current_context, machine, metadata=None,
                   policy="balanced", observed=None, kv=None, kv_k=None,
                   kv_v=None, batch=None, ubatch=None,
                   fit_target=None, no_fit=False, no_cache_reuse=False,
                   ceiling=None):
    """Pick a context and runtime together, returning a TunedChoice."""
    overrides = {}
    if kv is not None:
        overrides["kv"] = kv
    if kv_k is not None:
        overrides["kv_k"] = kv_k
    if kv_v is not None:
        overrides["kv_v"] = kv_v
    if batch is not None:
        overrides["batch"] = batch
    if ubatch is not None:
        overrides["ubatch"] = ubatch
    if fit_target is not None:
        overrides["fit_target"] = fit_target
    if no_fit:
        overrides["no_fit"] = True
    if no_cache_reuse:
        overrides["no_cache_reuse"] = True

    runtime = policy_runtime(policy, overrides)
    candidates = candidates_for(policy, observed)
    choice = hardware.choose_context(
        model_size, current_context, machine, metadata,
        candidates=candidates, ceiling=ceiling,
        kv_type_k=runtime["kv_type_k"], kv_type_v=runtime["kv_type_v"])
    return TunedChoice(
        context=choice.context,
        runtime=runtime,
        reason=choice.reason,
        required=choice.required,
        budget=choice.budget,
        approximate=choice.approximate,
        native=choice.native,
    )

"""Shared llama.cpp model profiles and state paths for llm."""
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

HOME = Path.home()
STATE = Path(os.environ.get(
    "LMM_STATE_DIR", HOME / ".local" / "state" / "llm")).expanduser()
LEGACY_STATE = Path(os.environ.get(
    "LMM_LEGACY_STATE_DIR", HOME / ".local" / "state" / "graphify-model-service")).expanduser()
URL = os.environ.get("LMM_URL", "http://127.0.0.1:18081").rstrip("/")
PORT = int(os.environ.get("LMM_PORT", "18081"))
SELECTION = STATE / "selected-model.json"
DEFAULT = "qwen2.5-7b-instruct-q4-k-m"

BUILTIN_PROFILES = {
    DEFAULT: {
        "id": DEFAULT,
        "alias": "qwen2.5-7b-instruct-graphify",
        "path": str(HOME / ".local" / "share" / "graphify-qwen2.5-7b-instruct"
                    / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
        "context": 8192,
        "sampling": {"temperature": 0, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0},
    },
    "glm-4.7-flash-ud-q4-k-xl": {
        "id": "glm-4.7-flash-ud-q4-k-xl",
        "alias": "glm-4.7-flash",
        "hf": "unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL",
        "path": str(HOME / ".local" / "share" / "graphify-glm-4.7-flash"
                    / "GLM-4.7-Flash-UD-Q4_K_XL.gguf"),
        "context": 32768,
        "sampling": {"temperature": 0.7, "top_p": 1.0, "min_p": 0.01, "repeat_penalty": 1.0},
    },
    "inkling-2-mini-gguf-ud-q4-k-xl": {
        "id": "inkling-2-mini-gguf-ud-q4-k-xl",
        "alias": "inkling-2-mini",
        "path": str(HOME / ".local" / "share" / "graphify-inkling-2-mini"
                    / "Inkling-2-Mini-UD-Q4_K_XL.gguf"),
        "context": 8192,
        "sampling": {"temperature": 0.7, "top_p": 1.0, "min_p": 0.01, "repeat_penalty": 1.0},
        "min_llama_cpp": "1.1.0",
    },
}


class ModelError(ValueError):
    def __init__(self, message, candidates=()):
        super().__init__(message)
        self.candidates = list(candidates)

USER_MODELS = STATE / "models.json"


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise ModelError("Invalid model profile: expected an object")
    missing = [key for key in ("id", "alias", "path", "context", "sampling")
               if key not in profile]
    if missing:
        raise ModelError("Invalid model profile %r; missing: %s"
                         % (profile.get("id", "?"), ", ".join(missing)))


def load_user_profiles(path=None):
    path = USER_MODELS if path is None else Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("models"), dict):
        data = data["models"]
    profiles = {}
    for model_id, profile in data.items():
        if not isinstance(profile, dict):
            continue
        profile = dict(profile)
        profile.setdefault("id", model_id)
        try:
            validate_profile(profile)
        except ModelError:
            continue
        profiles[profile["id"]] = profile
    return profiles


def load_profiles(path=None, builtin=None):
    source = BUILTIN_PROFILES if builtin is None else builtin
    merged = {model_id: dict(profile) for model_id, profile in source.items()}
    aliases = {str(profile.get("alias") or "").strip().lower()
               for profile in merged.values()}
    aliases.discard("")
    for model_id, profile in load_user_profiles(path).items():
        if model_id in merged:
            continue
        alias = str(profile.get("alias") or "").strip().lower()
        if alias and alias in aliases:
            continue
        merged[model_id] = profile
        if alias:
            aliases.add(alias)
    return merged


PROFILES = load_profiles()


def register_model(profile, path=None, profiles=None):
    validate_profile(profile)
    profiles = PROFILES if profiles is None else profiles
    model_id = profile["id"]
    if model_id in BUILTIN_PROFILES:
        raise ModelError("Refusing to overwrite built-in model %r" % model_id)
    for other_id, other in profiles.items():
        if (other_id != model_id
                and str(other.get("alias") or "").strip().lower()
                == profile["alias"].strip().lower()):
            raise ModelError("Alias %r is already used by %s"
                             % (profile["alias"], other_id))
    registry_path = USER_MODELS if path is None else Path(path)
    registry = load_user_profiles(registry_path)
    registry[model_id] = profile
    registry_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=registry_path.parent,
                                         delete=False) as out:
            temporary = Path(out.name)
            json.dump(registry, out, indent=2, sort_keys=True)
        os.replace(temporary, registry_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    profiles[model_id] = profile
    return profile


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalize_alias(name):
    return _slug(name)


def _unique(base, taken):
    candidate = base or "model"
    suffix = 2
    while candidate in taken:
        candidate = "%s-%d" % (base, suffix)
        suffix += 1
    return candidate


def derive_model_names(repo, filename, taken_ids, taken_aliases):
    repo_part = _slug(repo.replace("/", "-"))
    stem = Path(filename).stem
    model_id = _unique(_slug(repo_part + "-" + stem), set(taken_ids))
    alias = _unique(_slug(stem), set(taken_aliases))
    return model_id, alias


def _names(profile):
    return (profile.get("id", ""), profile.get("alias", ""))


def resolve_model(name, profiles=None):
    """Resolve an id or alias, then a unique prefix, then a unique substring."""
    profiles = PROFILES if profiles is None else profiles
    query = name.strip().lower()
    if not query:
        raise ModelError("No model name given; choose one of: " + ", ".join(sorted(profiles)),
                         candidates=sorted(profiles))

    exact = [pid for pid, profile in profiles.items()
             if query in (str(pid).lower(), profile.get("alias", "").lower())]
    if len(exact) == 1:
        return profiles[exact[0]]

    prefix = [pid for pid, profile in profiles.items()
              if any(value.lower().startswith(query) for value in _names(profile))]
    matches = prefix or [pid for pid, profile in profiles.items()
                         if any(query in value.lower() for value in _names(profile))]
    if len(matches) == 1:
        return profiles[matches[0]]
    if not matches:
        raise ModelError(
            "Unknown model %r; choose one of: %s" % (name, ", ".join(sorted(profiles))),
            candidates=sorted(profiles))
    raise ModelError(
        "Ambiguous model %r; matches: %s" % (name, ", ".join(sorted(matches))),
        candidates=sorted(matches))


def selected_profile(selection=None, profiles=None):
    profiles = PROFILES if profiles is None else profiles
    selection = SELECTION if selection is None else selection
    try:
        name = json.loads(Path(selection).read_text())["model"]
    except FileNotFoundError:
        name = DEFAULT
    if name not in profiles:
        raise ModelError(
            "Unknown selected model %r; choose one of: %s"
            % (name, ", ".join(sorted(profiles))), candidates=sorted(profiles))
    return profiles[name]


def select_profile(name, selection=None, profiles=None):
    profiles = PROFILES if profiles is None else profiles
    selection = SELECTION if selection is None else Path(selection)
    profile = resolve_model(name, profiles=profiles)
    selection.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=selection.parent, delete=False) as out:
            temporary = Path(out.name)
            json.dump({"model": profile["id"]}, out)
        os.replace(temporary, selection)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return profile


def cache_directory(profile):
    # Include sampling/context so profile changes cannot replay another setup's labels.
    signature = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16]
    return STATE / "label-cache-v3" / profile["id"] / signature


def migrate_state():
    """Move legacy state (except the log) into the new state dir once."""
    if STATE.exists() or not LEGACY_STATE.exists():
        return False
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    for child in LEGACY_STATE.iterdir():
        if child.name == "server.log":
            continue
        shutil.move(str(child), str(STATE / child.name))
    return True

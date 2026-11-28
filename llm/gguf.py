"""Read GGUF metadata and estimate llama.cpp KV cache size."""
import struct

MAX_METADATA = 100_000
MAX_ARRAY = 100_000
MAX_STRING = 1 << 20

_TYPE_STRING = 8
_TYPE_ARRAY = 9

_SCALAR_FORMATS = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    7: "<?",   # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_SCALAR_SIZES = {key: struct.calcsize(fmt)
                 for key, fmt in _SCALAR_FORMATS.items()}


def _read_string(handle):
    length = struct.unpack("<Q", handle.read(8))[0]
    if length > MAX_STRING:
        raise ValueError("string too long")
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("truncated string")
    return data.decode("utf-8")


def _skip_string(handle):
    length = struct.unpack("<Q", handle.read(8))[0]
    if length > MAX_STRING:
        raise ValueError("string too long")
    handle.seek(length, 1)


def _read_array(handle, depth):
    if depth > 4:
        raise ValueError("array nesting too deep")
    item_type = struct.unpack("<I", handle.read(4))[0]
    count = struct.unpack("<Q", handle.read(8))[0]
    fmt = _SCALAR_FORMATS.get(item_type)
    if fmt is not None:
        size = _SCALAR_SIZES[item_type]
        if count > MAX_ARRAY:
            handle.seek(count * size, 1)
            return None
        data = handle.read(count * size)
        if len(data) != count * size:
            raise ValueError("truncated array")
        if not count:
            return []
        return list(struct.unpack("<%d%s" % (count, fmt[1:]), data))
    if item_type == _TYPE_STRING:
        for _ in range(count):
            _skip_string(handle)
        return None
    if item_type == _TYPE_ARRAY:
        for _ in range(count):
            _read_array(handle, depth + 1)
        return None
    raise ValueError("unknown array type %d" % item_type)


def _read_value(handle, value_type, depth):
    if value_type == _TYPE_STRING:
        return _read_string(handle)
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is not None:
        return struct.unpack(
            fmt, handle.read(_SCALAR_SIZES[value_type]))[0]
    if value_type == _TYPE_ARRAY:
        return _read_array(handle, depth)
    raise ValueError("unknown type %d" % value_type)


def read_metadata(path):
    """Return the GGUF metadata mapping, or None if the file is unusable."""
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return None
            version, tensor_count, kv_count = struct.unpack(
                "<IQQ", handle.read(20))
            if kv_count > MAX_METADATA:
                return None
            metadata = {}
            for _ in range(kv_count):
                key = _read_string(handle)
                value_type = struct.unpack("<I", handle.read(4))[0]
                metadata[key] = _read_value(handle, value_type, 0)
            return metadata
    except (struct.error, OSError, UnicodeDecodeError, ValueError):
        return None


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _positive_int_list(value, length):
    if not isinstance(value, list) or len(value) != length:
        return None
    items = [_positive_int(item) for item in value]
    if any(item is None for item in items):
        return None
    return items


KV_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
}


def kv_cache_bytes(metadata, context, kv_type_k="f16", kv_type_v="f16"):
    """Estimate KV cache bytes for a context, or None if it cannot."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    if _positive_int(context) is None:
        return None
    try:
        bytes_k = KV_BYTES_PER_ELEMENT[kv_type_k]
        bytes_v = KV_BYTES_PER_ELEMENT[kv_type_v]
    except KeyError:
        return None
    n_layer = _positive_int(metadata.get(arch + ".block_count"))
    if n_layer is None:
        return None
    head_raw = metadata.get(arch + ".attention.head_count")
    if isinstance(head_raw, list):
        head_counts = _positive_int_list(head_raw, n_layer)
    else:
        head_counts = _positive_int(head_raw)
    if head_counts is None:
        return None
    kv_raw = metadata.get(arch + ".attention.head_count_kv", head_raw)
    if isinstance(kv_raw, list):
        kv_counts = _positive_int_list(kv_raw, n_layer)
    else:
        kv_count = _positive_int(kv_raw)
        kv_counts = None if kv_count is None else [kv_count] * n_layer
    if kv_counts is None:
        return None
    d_key = _positive_int(metadata.get(arch + ".attention.key_length"))
    d_value = _positive_int(metadata.get(arch + ".attention.value_length"))
    if d_key is None or d_value is None:
        embd = _positive_int(metadata.get(arch + ".embedding_length"))
        if embd is None:
            return None
    if isinstance(head_counts, list):
        per_layer_heads = head_counts
    else:
        per_layer_heads = [head_counts] * n_layer
    total = 0
    for index, heads in enumerate(per_layer_heads):
        key_dim = d_key if d_key is not None else embd // heads
        value_dim = d_value if d_value is not None else embd // heads
        if key_dim <= 0 or value_dim <= 0:
            return None
        total += kv_counts[index] * (
            key_dim * bytes_k + value_dim * bytes_v)
    if total <= 0:
        return None
    return int(context * total)


def native_context(metadata):
    """Return the model's advertised context length, or None."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    return _positive_int(metadata.get(arch + ".context_length"))

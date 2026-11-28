import struct
import tempfile
import unittest
from pathlib import Path

from llm import gguf

META = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.embedding_length": 3584,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.context_length": 32768,
}

INKLING_META = {
    "general.architecture": "inkling2_loop",
    "inkling2_loop.block_count": 24,
    "inkling2_loop.embedding_length": 2304,
    "inkling2_loop.attention.head_count": [24] * 24,
    "inkling2_loop.attention.head_count_kv": [8] * 24,
    "inkling2_loop.context_length": 1048576,
}


def _string_bytes(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _encode(value):
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, str):
        return 8, _string_bytes(value)
    if isinstance(value, int):
        return 11, struct.pack("<q", value)
    if isinstance(value, float):
        return 12, struct.pack("<d", value)
    if isinstance(value, list):
        if not value:
            return 9, struct.pack("<IQ", 11, 0)
        item_type, payload = _encode(value[0])
        body = bytearray(payload)
        for item in value[1:]:
            other_type, other_payload = _encode(item)
            assert other_type == item_type
            body.extend(other_payload)
        return 9, struct.pack("<IQ", item_type, len(value)) + bytes(body)
    raise TypeError(value)


def write_gguf(path, metadata):
    buf = bytearray(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value in metadata.items():
        buf.extend(_string_bytes(key))
        value_type, payload = _encode(value)
        buf.extend(struct.pack("<I", value_type))
        buf.extend(payload)
    Path(path).write_bytes(bytes(buf))


class GgufTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write(self, metadata):
        path = self.dir / "model.gguf"
        write_gguf(path, metadata)
        return path

    def test_reads_architecture_and_headers(self):
        metadata = gguf.read_metadata(self._write(META))
        self.assertEqual(metadata["general.architecture"], "qwen2")
        self.assertEqual(metadata["qwen2.block_count"], 28)

    def test_kv_cache_uses_gqa_and_head_dim(self):
        metadata = gguf.read_metadata(self._write(META))
        self.assertEqual(gguf.kv_cache_bytes(metadata, 131072), 7_516_192_768)

    def test_kv_cache_sums_per_layer_array(self):
        meta = dict(META)
        meta["qwen2.attention.head_count_kv"] = [4] * 28
        metadata = gguf.read_metadata(self._write(meta))
        self.assertEqual(gguf.kv_cache_bytes(metadata, 131072), 7_516_192_768)

    def test_kv_cache_honours_explicit_key_and_value_length(self):
        meta = dict(META)
        meta["qwen2.attention.key_length"] = 64
        meta["qwen2.attention.value_length"] = 64
        metadata = gguf.read_metadata(self._write(meta))
        self.assertEqual(gguf.kv_cache_bytes(metadata, 131072), 3_758_096_384)

    def test_kv_cache_q8_0_is_half_f16(self):
        metadata = gguf.read_metadata(self._write(META))
        expected = int(7_516_192_768 * 1.0625 / 2.0)
        self.assertEqual(
            gguf.kv_cache_bytes(metadata, 131072, "q8_0", "q8_0"), expected)

    def test_kv_cache_q4_0_is_quarter_f16(self):
        metadata = gguf.read_metadata(self._write(META))
        expected = int(7_516_192_768 * 0.5625 / 2.0)
        self.assertEqual(
            gguf.kv_cache_bytes(metadata, 131072, "q4_0", "q4_0"), expected)

    def test_kv_cache_asymmetric_types(self):
        metadata = gguf.read_metadata(self._write(META))
        expected = 131072 * (28 * 4) * (128 * 1.0625 + 128 * 2.0)
        self.assertEqual(
            gguf.kv_cache_bytes(metadata, 131072, "q8_0", "f16"), expected)

    def test_kv_cache_handles_per_layer_head_counts(self):
        metadata = gguf.read_metadata(self._write(INKLING_META))
        self.assertEqual(gguf.kv_cache_bytes(metadata, 65536), 4_831_838_208)
        self.assertEqual(
            gguf.kv_cache_bytes(metadata, 65536, "q8_0", "q8_0"),
            2_566_914_048)
        self.assertEqual(
            gguf.kv_cache_bytes(metadata, 65536, "q4_0", "q4_0"),
            1_358_954_496)

    def test_kv_cache_list_heads_with_scalar_kv(self):
        meta = dict(META)
        meta["qwen2.attention.head_count"] = [28] * 28
        metadata = gguf.read_metadata(self._write(meta))
        self.assertEqual(gguf.kv_cache_bytes(metadata, 131072),
                         7_516_192_768)

    def test_kv_cache_mismatched_head_lists_return_none(self):
        meta = dict(META)
        meta["qwen2.attention.head_count"] = [28] * 27
        metadata = gguf.read_metadata(self._write(meta))
        self.assertIsNone(gguf.kv_cache_bytes(metadata, 131072))
        meta = dict(META)
        meta["qwen2.attention.head_count_kv"] = [4] * 27
        metadata = gguf.read_metadata(self._write(meta))
        self.assertIsNone(gguf.kv_cache_bytes(metadata, 131072))

    def test_kv_cache_unknown_type_returns_none(self):
        metadata = gguf.read_metadata(self._write(META))
        self.assertIsNone(
            gguf.kv_cache_bytes(metadata, 131072, "q3_k", "f16"))
        self.assertIsNone(
            gguf.kv_cache_bytes(metadata, 131072, "f16", "q3_k"))
        self.assertIsNone(gguf.kv_cache_bytes({}, 131072, "q8_0", "q8_0"))

    def test_missing_metadata_returns_none(self):
        self.assertIsNone(gguf.kv_cache_bytes({}, 131072))
        self.assertIsNone(
            gguf.kv_cache_bytes({"general.architecture": "qwen2"}, 131072))

    def test_truncated_file_returns_none(self):
        path = self.dir / "model.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 5)[:8])
        self.assertIsNone(gguf.read_metadata(path))

    def test_skips_large_string_arrays(self):
        meta = dict(META)
        meta["tokenizer.ggml.tokens"] = ["token-%d" % i for i in range(5000)]
        metadata = gguf.read_metadata(self._write(meta))
        self.assertEqual(metadata["general.architecture"], "qwen2")

    def test_native_context_reads_context_length(self):
        self.assertEqual(gguf.native_context(META), 32768)
        self.assertIsNone(gguf.native_context({}))


if __name__ == "__main__":
    unittest.main()

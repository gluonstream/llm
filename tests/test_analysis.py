import struct
import tempfile
import unittest
from pathlib import Path

from llm import analysis, hardware

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

GIB = 1024 ** 3


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


def _machine(total_ram, unified=True):
    return hardware.Hardware(
        chip="Apple M1", model="arm64", total_ram=total_ram, unified=unified,
        gpu="unknown", vram=None, disk_free=100 * GIB,
        available_ram=total_ram, gpus=[])


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _profile(self, model_id, metadata, alias=None):
        path = self.dir / (model_id + ".gguf")
        write_gguf(path, metadata)
        return {
            "id": model_id, "alias": alias or model_id, "path": str(path),
            "context": 8192, "sampling": {},
        }

    def test_rows_compute_three_kv_types(self):
        profiles = {"qwen": self._profile("qwen", META)}
        rows = analysis.rows(profiles, context=65536)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["model"], "qwen")
        self.assertEqual(row["f16"], 3_758_096_384)
        self.assertEqual(row["q8_0"], 1_996_488_704)
        self.assertEqual(row["q4_0"], 1_056_964_608)

    def test_rows_skip_missing_or_unreadable_metadata(self):
        garbage = self.dir / "garbage.gguf"
        garbage.write_bytes(b"not a gguf file")
        profiles = {
            "gone": {"id": "gone", "alias": "gone",
                     "path": str(self.dir / "does-not-exist.gguf"),
                     "context": 8192, "sampling": {}},
            "unreadable": {"id": "unreadable", "alias": "unreadable",
                           "path": str(garbage), "context": 8192,
                           "sampling": {}},
        }
        rows = analysis.rows(profiles, context=65536)
        self.assertEqual([row["model"] for row in rows], ["gone", "unreadable"])
        for row in rows:
            self.assertIsNone(row["f16"])
            self.assertIsNone(row["q8_0"])
            self.assertIsNone(row["q4_0"])
            self.assertIsNone(row["saving"])
            self.assertIsNone(row["fits"])
            self.assertIsNone(row["native"])
        self.assertIsNone(rows[0]["size"])
        self.assertIsNotNone(rows[1]["size"])

    def test_rows_handle_inkling_per_layer_heads(self):
        profiles = {"inkling": self._profile("inkling", INKLING_META)}
        rows = analysis.rows(profiles, context=65536)
        row = rows[0]
        self.assertEqual(row["f16"], 4_831_838_208)
        self.assertEqual(row["q8_0"], 2_566_914_048)
        self.assertEqual(row["q4_0"], 1_358_954_496)

    def test_rows_saving_and_native(self):
        profiles = {"qwen": self._profile("qwen", META)}
        row = analysis.rows(profiles, context=65536)[0]
        self.assertAlmostEqual(row["saving"], 0.46875)
        self.assertEqual(row["native"], 32768)
        self.assertIsNone(row["fits"])  # no machine

    def test_rows_fits_requires_size_and_machine(self):
        profiles = {"qwen": self._profile("qwen", META)}
        row = analysis.rows(profiles, context=65536,
                            machine=_machine(128 * GIB))[0]
        self.assertIs(row["fits"], True)
        row = analysis.rows(profiles, context=65536,
                            machine=_machine(2 * GIB))[0]
        self.assertIs(row["fits"], False)


if __name__ == "__main__":
    unittest.main()

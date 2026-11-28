import unittest

from llm import hardware, runtime
from tests.test_hardware import machine, model_metadata


class RuntimePolicyTest(unittest.TestCase):
    def test_balanced_policy_fields(self):
        policy = runtime.POLICIES["balanced"]
        self.assertEqual(policy["kv_type_k"], "q8_0")
        self.assertEqual(policy["kv_type_v"], "q8_0")
        self.assertEqual(policy["candidates"], [65536])
        self.assertEqual(policy["batch"], 1024)
        self.assertEqual(policy["ubatch"], 256)
        self.assertEqual(policy["cache_reuse"], 256)
        self.assertTrue(policy["fit"])
        self.assertEqual(policy["fit_target"], 2048)
        self.assertEqual(policy["flash_attn"], "on")
        self.assertEqual(policy["parallel"], 1)
        self.assertTrue(policy["cache_prompt"])
        self.assertTrue(policy["kv_unified"])
        self.assertFalse(policy["no_kv_offload"])

    def test_conservative_and_extended_kv(self):
        conservative = runtime.POLICIES["conservative"]
        self.assertEqual(conservative["kv_type_k"], "f16")
        self.assertEqual(conservative["kv_type_v"], "f16")
        self.assertEqual(conservative["candidates"], [65536])
        self.assertEqual(conservative["batch"], 1024)
        self.assertEqual(conservative["ubatch"], 256)
        extended = runtime.POLICIES["extended"]
        self.assertEqual(extended["kv_type_k"], "q8_0")
        self.assertEqual(extended["kv_type_v"], "q8_0")
        self.assertEqual(extended["candidates"], [131072])
        self.assertEqual(extended["batch"], 1024)
        self.assertEqual(extended["ubatch"], 256)

    def test_policy_runtime_omits_candidates(self):
        policy = runtime.policy_runtime("balanced")
        self.assertNotIn("candidates", policy)
        self.assertEqual(policy["kv_type_k"], "q8_0")

    def test_candidates_filter_by_observed_peak(self):
        self.assertEqual(runtime.candidates_for("balanced", 40_000), [65536])
        self.assertEqual(runtime.candidates_for("balanced", 70_000), [65536])
        # Peak beyond any candidate still returns the full list.
        self.assertEqual(runtime.candidates_for("balanced", 200_000), [65536])
        # No observation: unchanged.
        self.assertEqual(runtime.candidates_for("balanced"), [65536])

    def test_kv_override_applies_to_both(self):
        policy = runtime.policy_runtime("aggressive", {"kv": "q5_1"})
        self.assertEqual(policy["kv_type_k"], "q5_1")
        self.assertEqual(policy["kv_type_v"], "q5_1")

    def test_no_fit_and_no_cache_reuse(self):
        policy = runtime.policy_runtime("balanced", {
            "no_fit": True, "no_cache_reuse": True})
        self.assertFalse(policy["fit"])
        self.assertFalse(policy["cache_reuse"])

    def test_generation_and_speculative_defaults(self):
        policy = runtime.policy_runtime("balanced")
        self.assertEqual(policy["spec_type"], "none")
        self.assertEqual(policy["draft_model"], "")
        self.assertFalse(policy["cmoe"])
        self.assertEqual(policy["reasoning"], "auto")
        self.assertEqual(policy["reasoning_budget"], -1)
        self.assertEqual(policy["devices"], "")
        self.assertEqual(policy["top_k"], 40)
        self.assertEqual(policy["n_predict"], -1)

    def test_generation_and_speculative_overrides(self):
        choice = runtime.choose_runtime(
            12 * hardware.GIB, 8192, machine(40 * hardware.GIB),
            model_metadata(), policy="balanced",
            spec_type="draft-mtp", draft_model="/tmp/draft.gguf", cmoe=True,
            reasoning="off", reasoning_budget=1024, devices="0,1",
            top_k=80, n_predict=512)
        self.assertEqual(choice.runtime["spec_type"], "draft-mtp")
        self.assertEqual(choice.runtime["draft_model"], "/tmp/draft.gguf")
        self.assertTrue(choice.runtime["cmoe"])
        self.assertEqual(choice.runtime["reasoning"], "off")
        self.assertEqual(choice.runtime["reasoning_budget"], 1024)
        self.assertEqual(choice.runtime["devices"], "0,1")
        self.assertEqual(choice.runtime["top_k"], 80)
        self.assertEqual(choice.runtime["n_predict"], 512)

    def test_choose_runtime_uses_typed_kv(self):
        metadata = model_metadata()  # f16 KV == 1 GiB at 8192 context
        # Balanced (q8_0) fits 128K on 40 GiB; conservative (f16) would only
        # fit 64K because f16 KV is twice as large.
        machine_under_test = machine(40 * hardware.GIB)
        balanced = runtime.choose_runtime(
            12 * hardware.GIB, 8192, machine_under_test, metadata,
            policy="balanced")
        extended = runtime.choose_runtime(
            12 * hardware.GIB, 8192, machine_under_test, metadata,
            policy="extended")
        self.assertEqual(balanced.context, 65536)
        self.assertEqual(balanced.runtime["kv_type_k"], "q8_0")
        self.assertEqual(extended.context, 131072)
        self.assertEqual(extended.runtime["kv_type_k"], "q8_0")

    def test_balanced_downgrades_a_previous_128k_to_64k(self):
        metadata = model_metadata()
        choice = runtime.choose_runtime(
            12 * hardware.GIB, 131072, machine(40 * hardware.GIB), metadata,
            policy="balanced")
        self.assertEqual(choice.context, 65536)

    def test_explicit_lower_ceiling_is_respected(self):
        metadata = model_metadata()
        choice = runtime.choose_runtime(
            12 * hardware.GIB, 131072, machine(40 * hardware.GIB), metadata,
            policy="balanced", ceiling=32768)
        self.assertEqual(choice.context, 32768)

    def test_choose_runtime_keeps_current_when_none_fit(self):
        metadata = model_metadata()
        choice = runtime.choose_runtime(
            5 * hardware.GIB, 32768, machine(8 * hardware.GIB), metadata,
            policy="aggressive")
        self.assertEqual(choice.context, 32768)
        self.assertIn("no candidate fits", choice.reason)


if __name__ == "__main__":
    unittest.main()

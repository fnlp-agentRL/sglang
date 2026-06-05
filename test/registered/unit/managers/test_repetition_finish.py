"""Unit tests for Req._check_repetition_penalty_finish.

Covers the repetition-based early-stop path: the gate on repeat_min_count, the
suffix-only detection semantics, the min/max repeat-length knobs, the
FINISH_REPEAT reason it sets, and the incremental (append-only) contract the
RollingHashState relies on.
"""

import unittest

from sglang.srt.managers.schedule_batch import FINISH_REPEAT, Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestRepetitionFinish(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _make_req(self, **repeat_kwargs) -> Req:
        params = SamplingParams(max_new_tokens=4096, **repeat_kwargs)
        return Req(
            rid="test",
            origin_input_text="",
            origin_input_ids=[0],
            sampling_params=params,
        )

    def _check(self, req: Req, output_ids):
        """Set the output and run the repetition check once."""
        req.output_ids = list(output_ids)
        return req._check_repetition_penalty_finish()

    # ---- gating ----

    def test_disabled_by_default(self):
        req = self._make_req()  # repeat_min_count defaults to 0
        self.assertIsNone(req.rolling_hash_state)
        self.assertFalse(self._check(req, [1, 1, 1, 1, 1]))
        self.assertIsNone(req.finished_reason)
        self.assertFalse(req.finished())

    def test_disabled_when_min_count_is_one(self):
        # min_count == 1 means "one occurrence" (no actual repetition); the
        # feature is gated on repeat_min_count > 1, so the state is not created.
        req = self._make_req(repeat_min_count=1)
        self.assertIsNone(req.rolling_hash_state)

    # ---- detection ----

    def test_no_repeat_returns_false(self):
        req = self._make_req(repeat_min_count=3)
        self.assertFalse(self._check(req, [1, 2, 3, 4, 5]))
        self.assertIsNone(req.finished_reason)
        self.assertFalse(req.finished())

    def test_detects_suffix_repeat(self):
        req = self._make_req(repeat_min_count=3)
        # [5, 6] repeated 3 times at the end.
        self.assertTrue(self._check(req, [9, 5, 6, 5, 6, 5, 6]))
        self.assertIsInstance(req.finished_reason, FINISH_REPEAT)
        self.assertEqual(req.finished_reason.repeat_length, 2)
        self.assertTrue(req.finished())
        self.assertEqual(
            req.finished_reason.to_json(),
            {"type": "repeat", "repeat_length": 2},
        )

    def test_single_token_repeat(self):
        req = self._make_req(repeat_min_count=3)
        self.assertTrue(self._check(req, [1, 7, 7, 7]))
        self.assertEqual(req.finished_reason.repeat_length, 1)

    def test_repeat_must_reach_the_suffix(self):
        # [5, 6] x3 occurs but a trailing token breaks the suffix -> not detected.
        req = self._make_req(repeat_min_count=3)
        self.assertFalse(self._check(req, [5, 6, 5, 6, 5, 6, 9]))
        self.assertIsNone(req.finished_reason)

    def test_count_must_be_reached(self):
        # Only 2 consecutive copies, but 3 are required.
        req = self._make_req(repeat_min_count=3)
        self.assertFalse(self._check(req, [5, 6, 5, 6]))

    def test_min_length_filters_short_blocks(self):
        # [7] x4 would match at block length 1, but min length is 2 and a
        # length-2 block would need 6 tokens to appear 3 times.
        req = self._make_req(repeat_min_count=3, repeat_min_length=2)
        self.assertFalse(self._check(req, [7, 7, 7, 7]))
        # A genuine length-2 repeat is still caught.
        req = self._make_req(repeat_min_count=3, repeat_min_length=2)
        self.assertTrue(self._check(req, [1, 2, 1, 2, 1, 2]))
        self.assertEqual(req.finished_reason.repeat_length, 2)

    def test_max_length_caps_block(self):
        # True period is 3; capping the block length at 2 misses it (the
        # documented speed/recall trade-off).
        req = self._make_req(repeat_min_count=3, repeat_max_length=2)
        self.assertFalse(self._check(req, [1, 2, 3, 1, 2, 3, 1, 2, 3]))
        # Allowing length 3 catches it.
        req = self._make_req(repeat_min_count=3, repeat_max_length=3)
        self.assertTrue(self._check(req, [1, 2, 3, 1, 2, 3, 1, 2, 3]))
        self.assertEqual(req.finished_reason.repeat_length, 3)

    # ---- incremental (append-only) usage, mirroring real decoding ----

    def test_incremental_decode_fires_when_repeat_completes(self):
        req = self._make_req(repeat_min_count=3, repeat_max_length=8)
        stream = [1, 2, 7, 7, 7]  # the 3rd '7' completes [7] x3
        fired_step = None
        for step, token in enumerate(stream):
            req.output_ids.append(token)
            if req._check_repetition_penalty_finish():
                fired_step = step
                break
        self.assertEqual(fired_step, 4)
        self.assertIsInstance(req.finished_reason, FINISH_REPEAT)
        self.assertEqual(req.finished_reason.repeat_length, 1)

    def test_incremental_long_run_with_window_sliding(self):
        # Long non-repeating prefix forces the RollingHashState window to slide
        # (start advances); the trailing repeat must still be detected.
        req = self._make_req(repeat_min_count=3, repeat_max_length=4)
        fired = False
        for i in range(200):
            req.output_ids.append(i)  # all distinct -> no repeat
            self.assertFalse(req._check_repetition_penalty_finish())
        self.assertGreater(req.rolling_hash_state.start, 0)  # window slid
        for token in [8, 9, 8, 9, 8, 9]:  # [8, 9] x3 at the very end
            req.output_ids.append(token)
            fired = req._check_repetition_penalty_finish()
        self.assertTrue(fired)
        self.assertEqual(req.finished_reason.repeat_length, 2)

    def test_retract_with_input_embeds_resets_state(self):
        # When an input_embeds request is retracted, output_ids is discarded.
        # The rolling hash state must be rebuilt; otherwise its stale
        # current_length outlives the output it described and the next check
        # either trips the growth assert or reads stale hashes.
        req = self._make_req(repeat_min_count=3)
        req.input_embeds = [[0.0]]  # mark as an input_embeds request

        for token in [1, 2, 3, 4, 5]:
            req.output_ids.append(token)
            self.assertFalse(req._check_repetition_penalty_finish())
        self.assertEqual(req.rolling_hash_state.current_length, 5)

        req.reset_for_retract()

        self.assertEqual(req.output_ids, [])
        self.assertEqual(req.rolling_hash_state.current_length, 0)
        self.assertEqual(req.rolling_hash_state.start, 0)

        # Re-generation starts clean: no stale-state crash, detection still works.
        fired = False
        for token in [8, 9, 8, 9, 8, 9]:  # [8, 9] x3
            req.output_ids.append(token)
            fired = req._check_repetition_penalty_finish()
        self.assertTrue(fired)
        self.assertEqual(req.finished_reason.repeat_length, 2)

    def test_assert_requires_output_growth(self):
        # has_repeat extends the rolling hash incrementally, so each call must
        # see at least one new token. Re-checking at the same length is a bug
        # the assert guards against.
        req = self._make_req(repeat_min_count=3)
        req.output_ids = [1, 2, 3]
        req._check_repetition_penalty_finish()  # advances current_length to 3
        with self.assertRaises(AssertionError):
            req._check_repetition_penalty_finish()


if __name__ == "__main__":
    unittest.main()

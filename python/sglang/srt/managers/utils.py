from __future__ import annotations

import dataclasses
import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import torch

from sglang.srt.eplb.expert_distribution import ExpertDistributionMetrics
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.server_args import ServerArgs
from sglang.srt.state_capturer.base import TopkCaptureOutput

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    num_correct_drafts: int = 0  # no bonus included
    num_correct_drafts_per_req_cpu: Optional[List[int]] = None
    can_run_cuda_graph: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[FutureIndices] = None
    speculative_num_draft_tokens: Optional[int] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    # Routed experts: pending async D2H for overlap scheduling
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None

    # metrics
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None

    # Forward pass metrics (FPM) — GPU-accurate timing via CUDA events
    fpm_start_event: Optional[torch.cuda.Event] = None
    fpm_end_event: Optional[torch.cuda.Event] = None

    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = (
                    self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.next_token_top_logprobs_val is not None:
                self.logits_output.next_token_top_logprobs_val = [
                    v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_top_logprobs_val
                ]
            if self.logits_output.next_token_top_logprobs_idx is not None:
                self.logits_output.next_token_top_logprobs_idx = [
                    x.to("cpu", non_blocking=True) if torch.is_tensor(x) else x
                    for x in self.logits_output.next_token_top_logprobs_idx
                ]
            if self.logits_output.next_token_token_ids_logprobs_val is not None:
                self.logits_output.next_token_token_ids_logprobs_val = [
                    v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_token_ids_logprobs_val
                ]
        if return_hidden_states and self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        if self.routed_experts_output is not None:
            self.routed_experts_output.copy_to_cpu()

        if self.indexer_topk_output is not None:
            self.indexer_topk_output.copy_to_cpu()

        if (x := self.expert_distribution_metrics) is not None:
            x.copy_to_cpu()

        self.copy_done.record()

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # TODO(lsyin): refactor PP and avoid using dict
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec v1:
    # 1) alloc topk * num_steps when draft decoding and then restore the allocation
    # 2) alloc num_draft_tokens when verifying the drafts
    # Sepc v2: allocate max(topk * num_steps, num_draft_tokens)

    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.speculative_num_draft_tokens
    page_size = server_args.page_size

    if page_size == 1 or spec_topk == 1:
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        raise NotImplementedError(
            "get_alloc_len_per_decode not implemented for page_size > 1 and spec_topk > 1"
        )


BASES = (131, 137)
MODS = ((1 << 61) - 1, (1 << 31) - 1)
POWER_TABLES = [array("q", [1]) for _ in MODS]  # post-mod powers, fit "q"


class RollingHashState:
    def __init__(
        self,
        min_count: int,
        min_repeat_length: int = 1,
        max_repeat_length: int | None = None,
    ) -> None:
        # Signed 64-bit array: all stored values are post-mod, in [0, 2**61 - 1),
        # so they fit "q" and use ~8 bytes each (vs ~28 + a pointer for boxed
        # ints in a list), giving lower memory and better cache locality.
        self.prefix_hashes = [array("q", [0]) for _ in MODS]  # length n + 1
        self.current_length = 0
        self.min_count = min_count
        self.min_repeat_length = min_repeat_length
        self.max_repeat_length = max_repeat_length
        self.start = 0
        self.window_size = (self.max_repeat_length or float("inf")) * self.min_count

    def extend(self, token_ids: Sequence[int], new_length: int) -> None:
        for hash_values, base, mod in zip(self.prefix_hashes, BASES, MODS):
            for i in range(self.current_length, new_length):
                hash_values.append((hash_values[-1] * base + token_ids[i] + 1) % mod)
        self.current_length = new_length

        stored_length = self.current_length - self.start

        if stored_length > self.window_size * 2 + 1:
            for hash_values in self.prefix_hashes:
                del hash_values[: self.window_size]
            self.start += self.window_size

    @staticmethod
    def grow_powers(new_length: int) -> None:
        for power_table, base, mod in zip(POWER_TABLES, BASES, MODS):
            while len(power_table) <= new_length:
                power_table.append(power_table[-1] * base % mod)

    def _equal_substrings(
        self, token_ids: Sequence[int], a: int, b: int, length: int
    ) -> bool:
        """Whether token_ids[a:a+length] == token_ids[b:b+length], via rolling hash."""
        if length == 0:
            return True
        probe = min(3, length)
        if any(token_ids[a + i] != token_ids[b + i] for i in range(probe)):
            return False
        a -= self.start
        b -= self.start
        for prefix_hashes, power_table, mod in zip(
            self.prefix_hashes, POWER_TABLES, MODS
        ):
            power = power_table[length]
            ha = (prefix_hashes[a + length] - prefix_hashes[a] * power) % mod
            hb = (prefix_hashes[b + length] - prefix_hashes[b] * power) % mod
            if ha != hb:
                return False
        return True

    def has_repeat(
        self,
        token_ids: Sequence[int],
    ) -> int:
        self.extend(token_ids, len(token_ids))
        self.grow_powers(min(len(token_ids), self.window_size * 2 + 1))

        upper = len(token_ids) // self.min_count
        if self.max_repeat_length is not None:
            upper = min(upper, self.max_repeat_length)

        for length in range(self.min_repeat_length, upper + 1):
            last = len(token_ids) - length  # absolute start of the final block
            if all(
                self._equal_substrings(
                    token_ids, last, len(token_ids) - j * length, length
                )
                for j in range(2, self.min_count + 1)
            ):
                return length
        return 0

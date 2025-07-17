import torch
import os

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)

REASONING_BUDGET = int(os.environ.get("REASONING_BUDGET", -1))

class BatchedReasoningBudgetingPenalizer(_BatchedPenalizer):
    """
    Min new tokens penalizer penalizes tokens based on the length of the output.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        return REASONING_BUDGET > 0

    def _prepare(self):
        self.reasoning_budget = REASONING_BUDGET + 1
        self.end_of_reasoning_token_id = 151668 # Qwen3 
        self.surpress_token_ids = [151668, 151645, 151643]  # think_end, eos, stop

        self.len_output_tokens = torch.zeros(
            size=(len(self.orchestrator.reqs()),),
            dtype=torch.int32,
            device="cpu",
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.len_output_tokens += 1

    def _apply(self, logits: torch.Tensor):


        under_budget_logit_indices = torch.nonzero(self.len_output_tokens < self.reasoning_budget).flatten()
        at_budget_logit_indices = torch.nonzero(self.len_output_tokens == self.reasoning_budget).flatten()

        if under_budget_logit_indices.numel() > 0:
            logits[under_budget_logit_indices[:, None], self.surpress_token_ids] = float('-inf')

        if at_budget_logit_indices.numel() > 0:
            logits[at_budget_logit_indices] = float('-inf')
            logits[at_budget_logit_indices, self.end_of_reasoning_token_id] = 99999.9

        return

    def _filter(self, keep_indices: torch.Tensor):
        self.len_output_tokens = self.len_output_tokens[keep_indices.cpu()]

    def _merge(self, their: "BatchedReasoningBudgetingPenalizer"):
        assert self.reasoning_budget == their.reasoning_budget, "Now we don't support different reasoning budgets"
        self.len_output_tokens = torch.cat(
            [self.len_output_tokens, their.len_output_tokens], dim=0
        )

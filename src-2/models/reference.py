from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class ReferenceModel(nn.Module):
    """Frozen reference model for KL divergence computation.

    Identical architecture to the policy, but all parameters are frozen.
    Used to compute ref_log_probs for the KL penalty term.
    """

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token log probabilities (no gradient).

        Args:
            input_ids: (B, T) token ids of [prompt + response].
            attention_mask: (B, T).
            labels_mask: (B, T) boolean mask for response tokens.

        Returns:
            log_probs: (B, T-1) per-token log probabilities.
        """
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        if labels_mask is not None:
            token_log_probs = token_log_probs * labels_mask[:, 1:]

        return token_log_probs

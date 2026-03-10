from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class CriticModel(nn.Module):
    """Value head model for PPO.

    Loads the same backbone architecture as the policy, replaces the lm_head
    with a linear projection to a scalar value.
    """

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        hidden_size = self.model.config.hidden_size
        # Remove the language model head, add a value head
        self.model.lm_head = nn.Identity()
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token values.

        Args:
            input_ids: (B, T) token ids.
            attention_mask: (B, T).
            labels_mask: (B, T) boolean mask for response positions.

        Returns:
            values: (B, T) scalar values per position.
        """
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits  # After Identity, this is the raw hidden states

        values = self.value_head(hidden_states).squeeze(-1)

        if labels_mask is not None:
            values = values * labels_mask

        return values

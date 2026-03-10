from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class PolicyModel(nn.Module):
    """Actor model that wraps a HuggingFace causal LM.

    Provides two core interfaces:
    - forward(): compute per-token log probabilities for existing sequences
    - generate(): auto-regressively sample new responses given prompts
    """

    def __init__(self, model_name_or_path: str, tokenizer_name: Optional[str] = None):
        super().__init__()
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token log probabilities.

        Args:
            input_ids: (B, T) token ids of [prompt + response].
            attention_mask: (B, T) attention mask.
            labels_mask: (B, T) boolean mask indicating which positions are
                response tokens. If None, all positions after the first are used.

        Returns:
            log_probs: (B, T-1) per-token log probabilities aligned with
                input_ids[:, 1:]. Positions outside labels_mask are zeroed.
        """
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        # Shift: logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        if labels_mask is not None:
            token_log_probs = token_log_probs * labels_mask[:, 1:]

        return token_log_probs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate response tokens given prompt ids.

        Returns:
            Full sequence ids (B, T_prompt + T_response).
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

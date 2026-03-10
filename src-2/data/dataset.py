from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer


class PromptDataset(Dataset):
    """Simple dataset that holds a list of prompt strings.

    Can be constructed from:
    - A list of strings
    - A HuggingFace datasets object (expects a 'prompt' column)
    - A JSONL file path
    """

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]

    @classmethod
    def from_hf_dataset(cls, dataset_name: str, split: str = "train", column: str = "prompt") -> "PromptDataset":
        """Load prompts from a HuggingFace dataset."""
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split)
        prompts = [row[column] for row in ds]
        return cls(prompts)

    @classmethod
    def from_jsonl(cls, path: str, column: str = "prompt") -> "PromptDataset":
        """Load prompts from a JSONL file."""
        import json
        prompts = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data[column])
        return cls(prompts)


class PromptCollator:
    """Collate function that tokenizes a batch of prompt strings.

    Returns padded input_ids and attention_mask tensors.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "prompt_ids": encoded["input_ids"],
            "prompt_mask": encoded["attention_mask"],
        }


def build_prompt_dataloader(
    dataset: PromptDataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    distributed: bool = True,
) -> DataLoader:
    """Build a DataLoader for prompts with optional distributed sampling."""
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    collator = PromptCollator(tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        drop_last=True,
    )

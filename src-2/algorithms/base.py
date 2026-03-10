from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import torch
    from src.rollout.generator import RolloutBatch


class RLAlgorithm(ABC):
    """Abstract base class for RL training algorithms."""

    @abstractmethod
    def compute_advantages(self, batch: RolloutBatch) -> RolloutBatch:
        """Compute advantage estimates and attach them to the batch.

        Args:
            batch: RolloutBatch from the rollout generator.

        Returns:
            The same batch, augmented with `advantages` and `returns` fields.
        """
        ...

    @abstractmethod
    def compute_loss(self, batch: RolloutBatch) -> Dict[str, torch.Tensor]:
        """Compute the training loss for a mini-batch.

        Returns:
            Dict of named loss components (e.g. policy_loss, value_loss, kl).
        """
        ...

    @abstractmethod
    def update_step(self, batch: RolloutBatch) -> Dict[str, float]:
        """Execute one full update step: advantages, mini-batch updates, logging.

        Args:
            batch: Complete RolloutBatch with all training signals.

        Returns:
            Dict of training statistics for logging.
        """
        ...

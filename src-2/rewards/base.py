from abc import ABC, abstractmethod
from typing import List


class RewardFunction(ABC):
    """Abstract base class for reward computation."""

    @abstractmethod
    def compute(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute reward scores for prompt-response pairs.

        Args:
            prompts: List of prompt strings.
            responses: List of response strings.

        Returns:
            List of scalar reward values, one per pair.
        """
        ...


class LengthReward(RewardFunction):
    """Simple reward based on response length.

    Useful for testing the training pipeline. Gives higher reward
    to responses within a target length range.
    """

    def __init__(self, target_length: int = 200, max_reward: float = 1.0):
        self.target_length = target_length
        self.max_reward = max_reward

    def compute(self, prompts: List[str], responses: List[str]) -> List[float]:
        rewards = []
        for resp in responses:
            length = len(resp.split())
            diff = abs(length - self.target_length)
            reward = self.max_reward * max(0.0, 1.0 - diff / self.target_length)
            rewards.append(reward)
        return rewards


class RuleBasedReward(RewardFunction):
    """Rule-based reward using keyword matching.

    Awards points for the presence of desired keywords and penalizes
    undesired patterns. Simple baseline for testing.
    """

    def __init__(
        self,
        positive_keywords: List[str] = None,
        negative_keywords: List[str] = None,
        positive_weight: float = 1.0,
        negative_weight: float = -1.0,
    ):
        self.positive_keywords = positive_keywords or []
        self.negative_keywords = negative_keywords or []
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def compute(self, prompts: List[str], responses: List[str]) -> List[float]:
        rewards = []
        for resp in responses:
            resp_lower = resp.lower()
            score = 0.0
            for kw in self.positive_keywords:
                if kw.lower() in resp_lower:
                    score += self.positive_weight
            for kw in self.negative_keywords:
                if kw.lower() in resp_lower:
                    score += self.negative_weight
            rewards.append(score)
        return rewards

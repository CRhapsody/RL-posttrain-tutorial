import re
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import sympy
from sympy.parsing.latex import parse_latex

logger = logging.getLogger("rl_trainer")


class RewardFunction(ABC):
    """Abstract base class for reward computation."""

    @abstractmethod
    def compute(self, prompts: List[str], responses: List[str]) -> List[float]:
        ...


# ─── Existing simple rewards (kept for pipeline testing) ─────────────


class LengthReward(RewardFunction):
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


# ─── RLVR: Math verification reward via sympy ────────────────────────


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_TAG_RE = re.compile(
    r"(?:The\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from model response.

    Tries in order:
      1. \\boxed{...} (LaTeX convention, e.g. DeepSeek-R1)
      2. "The answer is ..." / "The final answer is ..."
      3. Last number / expression in the response
    """
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]

    return None


def extract_ground_truth(prompt: str) -> Optional[str]:
    """Extract ground-truth answer embedded in the prompt.

    Expects the prompt to contain a tag like:
      <answer>42</answer>  or  [ANSWER: 3/4]  or  #### 42
    """
    patterns = [
        re.compile(r"<answer>\s*(.+?)\s*</answer>", re.IGNORECASE),
        re.compile(r"\[ANSWER:\s*(.+?)\]", re.IGNORECASE),
        re.compile(r"####\s*(.+)"),
    ]
    for pat in patterns:
        m = pat.search(prompt)
        if m:
            return m.group(1).strip()
    return None


def normalize_expr(text: str) -> Optional[sympy.Expr]:
    """Parse a string into a sympy expression for comparison.

    Handles: plain numbers, fractions (3/4), LaTeX (\\frac{3}{4}), symbols.
    """
    text = text.strip().rstrip(".")

    if not text:
        return None

    if "\\" in text or "frac" in text:
        try:
            return parse_latex(text)
        except Exception:
            pass

    text_clean = text.replace(" ", "")

    try:
        return sympy.sympify(text_clean, rational=True)
    except (sympy.SympifyError, SyntaxError, TypeError):
        pass

    return None


def sympy_equal(pred_str: str, gold_str: str) -> Optional[bool]:
    """Compare two math expressions using sympy.

    Returns True/False if comparison succeeds, None if parsing fails.
    """
    pred_expr = normalize_expr(pred_str)
    gold_expr = normalize_expr(gold_str)

    if pred_expr is None or gold_expr is None:
        return None

    try:
        diff = sympy.simplify(pred_expr - gold_expr)
        if diff == 0:
            return True
    except (TypeError, AttributeError):
        pass

    try:
        if pred_expr.equals(gold_expr):
            return True
    except (TypeError, AttributeError):
        pass

    try:
        pred_f = float(pred_expr.evalf())
        gold_f = float(gold_expr.evalf())
        if math.isclose(pred_f, gold_f, rel_tol=1e-6, abs_tol=1e-9):
            return True
    except (TypeError, ValueError):
        pass

    return False


class MathVerifyReward(RewardFunction):
    """RLVR-style reward: binary correctness via sympy verification.

    The prompt must contain a ground-truth answer in one of these formats:
      - <answer>42</answer>
      - [ANSWER: 3/4]
      - #### 42  (GSM8K style)

    The model response is parsed for the predicted answer (\\boxed{},
    "the answer is ...", or last number). The two are compared symbolically
    using sympy, awarding `correct_reward` for a match and
    `incorrect_reward` otherwise.

    Format reward (optional): adds a small bonus for well-structured
    responses that show reasoning steps.
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        format_reward: float = 0.0,
        unparseable_reward: float = 0.0,
    ):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.format_reward = format_reward
        self.unparseable_reward = unparseable_reward

    def _score_format(self, response: str) -> float:
        """Small bonus for structured responses (shows work + boxed answer)."""
        if self.format_reward == 0.0:
            return 0.0
        has_reasoning = any(
            kw in response.lower()
            for kw in ("step", "therefore", "thus", "so the answer", "we get", "=>")
        )
        has_boxed = _BOXED_RE.search(response) is not None
        if has_reasoning and has_boxed:
            return self.format_reward
        if has_reasoning or has_boxed:
            return self.format_reward * 0.5
        return 0.0

    def compute(self, prompts: List[str], responses: List[str]) -> List[float]:
        rewards = []
        for prompt, response in zip(prompts, responses):
            gold_str = extract_ground_truth(prompt)
            if gold_str is None:
                logger.warning("No ground truth found in prompt, giving 0 reward")
                rewards.append(self.unparseable_reward)
                continue

            pred_str = extract_answer(response)
            if pred_str is None:
                rewards.append(self.unparseable_reward + self._score_format(response))
                continue

            result = sympy_equal(pred_str, gold_str)

            if result is True:
                reward = self.correct_reward
            elif result is False:
                reward = self.incorrect_reward
            else:
                reward = self.unparseable_reward

            reward += self._score_format(response)
            rewards.append(reward)

        return rewards

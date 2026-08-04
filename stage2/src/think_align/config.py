from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextConfig:
    """Parsing and truncation settings."""

    max_generated_sentences: int = 8
    max_reference_sentences: int = 8
    max_words_per_sentence: int = 96
    split_on_semicolon: bool = True

    def __post_init__(self) -> None:
        if self.max_generated_sentences < 1:
            raise ValueError("max_generated_sentences must be >= 1")
        if self.max_reference_sentences < 1:
            raise ValueError("max_reference_sentences must be >= 1")
        if self.max_words_per_sentence < 2:
            raise ValueError("max_words_per_sentence must be >= 2")


@dataclass(frozen=True)
class OTConfig:
    """Semi-relaxed Sinkhorn and score-conversion settings."""

    entropy_reg: float = 0.05
    marginal_relaxation: float = 1.0
    score_temperature: float = 0.5
    max_iterations: int = 100
    tolerance: float = 1e-5
    cost_min: float = 0.0
    cost_max: float = 2.0
    mass_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.entropy_reg <= 0:
            raise ValueError("entropy_reg must be > 0")
        if self.marginal_relaxation <= 0:
            raise ValueError("marginal_relaxation must be > 0")
        if self.score_temperature <= 0:
            raise ValueError("score_temperature must be > 0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if self.cost_max <= self.cost_min:
            raise ValueError("cost_max must exceed cost_min")


@dataclass(frozen=True)
class ThinkAlignConfig:
    """Top-level ThinkAlign configuration."""

    text: TextConfig = field(default_factory=TextConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    contradiction_gamma: float = 1.0
    reward_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.contradiction_gamma < 0:
            raise ValueError("contradiction_gamma must be >= 0")
        if self.reward_epsilon <= 0:
            raise ValueError("reward_epsilon must be > 0")

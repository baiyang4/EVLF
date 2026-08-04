"""clinical reasoning alignment (r_think)."""

from .config import ThinkAlignConfig, OTConfig, TextConfig
from .encoder import (
    BaseTokenEncoder,
    EncodedSentence,
    FrozenHFTokenEncoder,
    IDFTokenWeighter,
    StaticTokenEncoder,
)
from .nli import (
    BaseContradictionScorer,
    FixedContradictionScorer,
    HFContradictionScorer,
    NullContradictionScorer,
)
from .reward import ThinkAlignDiagnostics, ThinkAlignSentenceAligner

__all__ = [
    "ThinkAlignConfig",
    "OTConfig",
    "TextConfig",
    "BaseTokenEncoder",
    "EncodedSentence",
    "FrozenHFTokenEncoder",
    "IDFTokenWeighter",
    "StaticTokenEncoder",
    "BaseContradictionScorer",
    "FixedContradictionScorer",
    "HFContradictionScorer",
    "NullContradictionScorer",
    "ThinkAlignDiagnostics",
    "ThinkAlignSentenceAligner",
]

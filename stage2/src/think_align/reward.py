from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ThinkAlignConfig
from .encoder import BaseTokenEncoder, EncodedSentence
from .nli import BaseContradictionScorer
from .ot import cosine_cost_matrix, distance_to_similarity, semi_relaxed_sinkhorn
from .text import split_sentences


@dataclass
class ThinkAlignDiagnostics:
    reward: float
    completeness: float
    faithfulness: float
    generated_sentences: list[str]
    reference_sentences: list[str]
    completeness_matrix: torch.Tensor
    faithfulness_matrix: torch.Tensor
    contradiction_matrix: torch.Tensor
    best_generated_for_reference: list[int]
    best_reference_for_generated: list[int]


class ThinkAlignSentenceAligner:
    """clinical reasoning alignment reward."""

    def __init__(
        self,
        token_encoder: BaseTokenEncoder,
        contradiction_scorer: BaseContradictionScorer,
        config: ThinkAlignConfig | None = None,
    ) -> None:
        self.token_encoder = token_encoder
        self.contradiction_scorer = contradiction_scorer
        self.config = config or ThinkAlignConfig()

    def _encode_sentences(self, sentences: list[str]) -> list[EncodedSentence]:
        return [
            self.token_encoder.encode(
                sentence,
                max_words=self.config.text.max_words_per_sentence,
            )
            for sentence in sentences
        ]

    @torch.no_grad()
    def score(
        self,
        generated_reasoning: str,
        reference_reasoning: str,
    ) -> ThinkAlignDiagnostics:
        generated_sentences = split_sentences(
            generated_reasoning,
            split_on_semicolon=self.config.text.split_on_semicolon,
            max_sentences=self.config.text.max_generated_sentences,
        )
        reference_sentences = split_sentences(
            reference_reasoning,
            split_on_semicolon=self.config.text.split_on_semicolon,
            max_sentences=self.config.text.max_reference_sentences,
        )

        if not generated_sentences or not reference_sentences:
            empty = torch.empty(
                (len(generated_sentences), len(reference_sentences)),
                dtype=torch.float32,
            )
            return ThinkAlignDiagnostics(
                reward=0.0,
                completeness=0.0,
                faithfulness=0.0,
                generated_sentences=generated_sentences,
                reference_sentences=reference_sentences,
                completeness_matrix=empty,
                faithfulness_matrix=empty.clone(),
                contradiction_matrix=empty.clone(),
                best_generated_for_reference=[],
                best_reference_for_generated=[],
            )

        gen_encoded = self._encode_sentences(generated_sentences)
        ref_encoded = self._encode_sentences(reference_sentences)

        # Score both NLI directions in one batched call.
        forward_pairs: list[tuple[str, str]] = []
        backward_pairs: list[tuple[str, str]] = []
        for generated in generated_sentences:
            for reference in reference_sentences:
                forward_pairs.append((generated, reference))
                backward_pairs.append((reference, generated))
        forward_contra = self.contradiction_scorer.score_pairs(forward_pairs)
        backward_contra = self.contradiction_scorer.score_pairs(backward_pairs)
        bidirectional_contra = torch.maximum(forward_contra, backward_contra)

        m = len(generated_sentences)
        n = len(reference_sentences)
        contradiction_matrix = bidirectional_contra.reshape(m, n)
        completeness_matrix = torch.zeros((m, n), dtype=torch.float32)
        faithfulness_matrix = torch.zeros((m, n), dtype=torch.float32)

        for i, generated in enumerate(gen_encoded):
            for j, reference in enumerate(ref_encoded):
                device = generated.embeddings.device
                reference_on_device = reference.to(device)
                cost = cosine_cost_matrix(
                    generated.embeddings,
                    reference_on_device.embeddings,
                    cost_min=self.config.ot.cost_min,
                    cost_max=self.config.ot.cost_max,
                )

                completeness_ot = semi_relaxed_sinkhorn(
                    cost,
                    generated.mass,
                    reference_on_device.mass,
                    hard_marginal="target",
                    entropy_reg=self.config.ot.entropy_reg,
                    marginal_relaxation=self.config.ot.marginal_relaxation,
                    max_iterations=self.config.ot.max_iterations,
                    tolerance=self.config.ot.tolerance,
                    mass_epsilon=self.config.ot.mass_epsilon,
                )
                faithfulness_ot = semi_relaxed_sinkhorn(
                    cost,
                    generated.mass,
                    reference_on_device.mass,
                    hard_marginal="source",
                    entropy_reg=self.config.ot.entropy_reg,
                    marginal_relaxation=self.config.ot.marginal_relaxation,
                    max_iterations=self.config.ot.max_iterations,
                    tolerance=self.config.ot.tolerance,
                    mass_epsilon=self.config.ot.mass_epsilon,
                )

                transport_completeness = distance_to_similarity(
                    completeness_ot.distance,
                    self.config.ot.score_temperature,
                )
                transport_faithfulness = distance_to_similarity(
                    faithfulness_ot.distance,
                    self.config.ot.score_temperature,
                )

                contradiction = contradiction_matrix[i, j].to(device)
                gate = (1.0 - contradiction.clamp(0.0, 1.0)).pow(
                    self.config.contradiction_gamma
                )
                completeness_matrix[i, j] = (
                    transport_completeness * gate
                ).cpu()
                faithfulness_matrix[i, j] = (
                    transport_faithfulness * gate
                ).cpu()

        completeness_per_reference, best_gen = completeness_matrix.max(dim=0)
        faithfulness_per_generated, best_ref = faithfulness_matrix.max(dim=1)
        completeness = completeness_per_reference.mean()
        faithfulness = faithfulness_per_generated.mean()
        reward = (
            2.0
            * completeness
            * faithfulness
            / (completeness + faithfulness + self.config.reward_epsilon)
        ).clamp(0.0, 1.0)

        return ThinkAlignDiagnostics(
            reward=float(reward),
            completeness=float(completeness),
            faithfulness=float(faithfulness),
            generated_sentences=generated_sentences,
            reference_sentences=reference_sentences,
            completeness_matrix=completeness_matrix,
            faithfulness_matrix=faithfulness_matrix,
            contradiction_matrix=contradiction_matrix,
            best_generated_for_reference=best_gen.tolist(),
            best_reference_for_generated=best_ref.tolist(),
        )

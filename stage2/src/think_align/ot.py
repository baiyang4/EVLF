from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

HardMarginal = Literal["source", "target"]


@dataclass
class SemiRelaxedOTResult:
    distance: torch.Tensor
    transport_cost: torch.Tensor
    marginal_penalty: torch.Tensor
    plan: torch.Tensor
    source_marginal: torch.Tensor
    target_marginal: torch.Tensor
    iterations: int
    converged: bool


def normalize_mass(mass: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mass.ndim != 1:
        raise ValueError("mass must be one-dimensional")
    if bool((mass < 0).any()):
        raise ValueError("mass cannot contain negative values")
    mass = mass.float().clamp_min(eps)
    return mass / mass.sum().clamp_min(eps)


def generalized_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Generalized KL: sum p log(p/q) - p + q."""

    p_safe = p.clamp_min(eps)
    q_safe = q.clamp_min(eps)
    return (p_safe * (p_safe.log() - q_safe.log()) - p_safe + q_safe).sum()


def cosine_cost_matrix(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    cost_min: float = 0.0,
    cost_max: float = 2.0,
) -> torch.Tensor:
    if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [tokens, hidden_dim]")
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("source and target hidden dimensions must match")
    source = F.normalize(source_embeddings.float(), dim=-1, eps=1e-8)
    target = F.normalize(target_embeddings.float(), dim=-1, eps=1e-8)
    cost = 1.0 - source @ target.transpose(0, 1)
    return cost.clamp(min=cost_min, max=cost_max)


@torch.no_grad()
def semi_relaxed_sinkhorn(
    cost: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    hard_marginal: HardMarginal,
    entropy_reg: float = 0.05,
    marginal_relaxation: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    mass_epsilon: float = 1e-8,
) -> SemiRelaxedOTResult:
    """Solve one-sided semi-relaxed entropic OT in the log domain.

    `hard_marginal="target"` is used for completeness: every reference-token
    mass must be covered, while generated-token mass may be redistributed under
    a KL penalty.

    `hard_marginal="source"` is used for faithfulness: every generated-token
    mass must be explained, while reference-token mass may be redistributed.

    The plan is found using generalized Sinkhorn updates. The reported distance
    excludes the entropy term and equals transport cost plus the soft-marginal
    KL penalty, reducing entropy-induced score bias.
    """

    if cost.ndim != 2:
        raise ValueError("cost must have shape [source_tokens, target_tokens]")
    if cost.shape != (source_mass.numel(), target_mass.numel()):
        raise ValueError("cost shape must match source and target mass lengths")
    if hard_marginal not in {"source", "target"}:
        raise ValueError("hard_marginal must be 'source' or 'target'")
    if entropy_reg <= 0 or marginal_relaxation <= 0:
        raise ValueError("entropy_reg and marginal_relaxation must be > 0")

    device = cost.device
    dtype = torch.float32
    cost = cost.to(device=device, dtype=dtype)
    a = normalize_mass(source_mass.to(device=device, dtype=dtype), mass_epsilon)
    b = normalize_mass(target_mass.to(device=device, dtype=dtype), mass_epsilon)
    log_a = a.clamp_min(mass_epsilon).log()
    log_b = b.clamp_min(mass_epsilon).log()
    log_kernel = -cost / entropy_reg

    # Exponent for the KL-relaxed marginal. A hard marginal uses exponent 1.
    rho = marginal_relaxation / (marginal_relaxation + entropy_reg)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    converged = False

    for iteration in range(1, max_iterations + 1):
        old_u = log_u
        old_v = log_v

        if hard_marginal == "target":
            # Soft source update, then exact target scaling.
            log_kv = torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_u = rho * (log_a - log_kv)
            log_ktu = torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
            log_v = log_b - log_ktu
        else:
            # Soft target update, then exact source scaling.
            log_ktu = torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
            log_v = rho * (log_b - log_ktu)
            log_kv = torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_u = log_a - log_kv

        delta = torch.maximum(
            (log_u - old_u).abs().max(),
            (log_v - old_v).abs().max(),
        )
        if float(delta) < tolerance:
            converged = True
            break

    # Re-impose the hard marginal after the final relaxed update.
    if hard_marginal == "target":
        log_ktu = torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        log_v = log_b - log_ktu
    else:
        log_kv = torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_u = log_a - log_kv

    log_plan = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
    plan = log_plan.clamp(min=-80.0, max=40.0).exp()
    source_marginal = plan.sum(dim=1)
    target_marginal = plan.sum(dim=0)

    transport_cost = (plan * cost).sum()
    if hard_marginal == "target":
        marginal_penalty = marginal_relaxation * generalized_kl(
            source_marginal, a, mass_epsilon
        )
    else:
        marginal_penalty = marginal_relaxation * generalized_kl(
            target_marginal, b, mass_epsilon
        )
    distance = transport_cost + marginal_penalty

    return SemiRelaxedOTResult(
        distance=distance,
        transport_cost=transport_cost,
        marginal_penalty=marginal_penalty,
        plan=plan,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        iterations=iteration,
        converged=converged,
    )


def distance_to_similarity(distance: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return torch.exp(-distance.clamp_min(0.0) / temperature).clamp(0.0, 1.0)

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    scorer_hidden_dim: int = 128
    order: int = 2
    alpha0: float = 1.0
    alpha1: float = 0.001
    alpha2: float = 0.001
    propagation_dropout: float = 0.5
    scorer_dropout: float = 0.2
    residual_gate_init: float = 0.5

    def __post_init__(self):
        for name in ("hidden_dim", "scorer_hidden_dim", "order"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("alpha0", "alpha1", "alpha2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("propagation_dropout", "scorer_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 < self.residual_gate_init < 1:
            raise ValueError("residual_gate_init must be in (0, 1)")


class MEE(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.drug_projection = nn.Linear(512, config.hidden_dim)
        self.target_projection = nn.Linear(1152, config.hidden_dim)
        self.dropout = nn.Dropout(config.propagation_dropout)

    def forward(self, drug_features, target_features):
        drug_hidden = self.drug_projection(drug_features)
        target_hidden = self.target_projection(target_features)
        h = torch.cat((drug_hidden, target_hidden), dim=0)
        return h, self.dropout(h)


class NASM(nn.Module):

    def __init__(self, num_nodes: int):
        super().__init__()
        self.beta = nn.Parameter(torch.full((num_nodes,), 1.0 / num_nodes))

    def forward(self, edge_index):
        n = self.beta.numel()
        b = F.relu(self.beta)
        degree = torch.bincount(edge_index[0], minlength=n).to(b.dtype) + b
        nonzero = degree > 0
        inv = torch.where(nonzero, degree, torch.ones_like(degree)).rsqrt()
        inv = inv * nonzero.to(b.dtype)
        row, col = edge_index
        edge_weights = inv[row] * inv[col]
        nodes = torch.arange(n, device=b.device)
        indices = torch.cat((edge_index, torch.stack((nodes, nodes))), dim=1)
        weights = torch.cat((edge_weights, b * inv.square()))
        return torch.sparse_coo_tensor(indices, weights, (n, n), check_invariants=True).coalesce()


class SRC(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.alpha0, self.alpha1 = config.alpha0, config.alpha1

    def forward(self, normalized):
        indices = normalized.indices()
        values = self.alpha0 * normalized.values()
        values = values + self.alpha1 * (indices[0] == indices[1]).to(values.dtype)
        return torch.sparse_coo_tensor(indices, values, normalized.shape, check_invariants=True).coalesce()


class ECMOPP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.alpha2 = config.alpha2
        self.gamma = nn.Parameter(torch.zeros(config.order + 1))
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.rho = nn.Parameter(torch.tensor(math.log(
            config.residual_gate_init / (1 - config.residual_gate_init))))
        self.scorer = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.scorer_hidden_dim),
            nn.GELU(), nn.Dropout(config.scorer_dropout),
            nn.Linear(config.scorer_hidden_dim, 2),
        )

    def edge_correction(self, h, edge_index):
        detached = h.detach()
        energy = detached.square().sum(0)
        safe_energy = torch.where(energy > 0, energy, torch.ones_like(energy))
        row, col = edge_index
        weights = (detached[row] * detached[col] / safe_energy).sum(1)
        return weights * (row != col).to(weights.dtype)

    def operator(self, structural, h):
        indices = structural.indices()
        values = structural.values() + self.alpha2 * self.edge_correction(h, indices)
        return torch.sparse_coo_tensor(indices, values, structural.shape, check_invariants=True).coalesce()

    def forward(self, structural, h, signal):
        matrix = self.operator(structural, h)
        pi = torch.softmax(self.gamma, dim=0)
        previous = signal
        current = torch.sparse.mm(matrix, previous)
        output = pi[0] * previous + pi[1] * current
        for k in range(2, len(pi)):
            following = 2 * torch.sparse.mm(matrix, current) - previous
            output = output + pi[k] * following
            previous, current = current, following
        gate = torch.sigmoid(self.rho)
        return (1 - gate) * h + gate * self.norm(output)

    def score(self, z, pairs, num_drugs):
        endpoints = torch.cat((z[pairs[:, 0]], z[num_drugs + pairs[:, 1]]), dim=1)
        return self.scorer(endpoints)


class NEASPDTI(nn.Module):
    def __init__(self, num_drugs: int, num_targets: int, config: ModelConfig | None = None):
        super().__init__()
        if num_drugs < 1 or num_targets < 1:
            raise ValueError("Both entity sets must be nonempty")
        self.config = config or ModelConfig()
        self.num_drugs, self.num_targets = num_drugs, num_targets
        self.mee = MEE(self.config)
        self.nasm = NASM(num_drugs + num_targets)
        self.src = SRC(self.config)
        self.ecmopp = ECMOPP(self.config)

    def encode(self, drug_features, target_features, edge_index):
        if drug_features.shape != (self.num_drugs, 512):
            raise ValueError("Expected drug features [num_drugs, 512]")
        if target_features.shape != (self.num_targets, 1152):
            raise ValueError("Expected target features [num_targets, 1152]")
        h, signal = self.mee(drug_features, target_features)
        return self.ecmopp(self.src(self.nasm(edge_index)), h, signal)

    def score(self, z, pairs):
        return self.ecmopp.score(z, pairs, self.num_drugs)

    def forward(self, drug_features, target_features, edge_index, pairs):
        return self.score(self.encode(drug_features, target_features, edge_index), pairs)


def multi_positive_infonce(drug_embeddings, target_embeddings, positive_pairs,
                          temperature: float = 0.2):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not len(positive_pairs):
        raise ValueError("InfoNCE requires positive training interactions")
    drugs = F.normalize(drug_embeddings, dim=1, eps=1e-12)
    targets = F.normalize(target_embeddings, dim=1, eps=1e-12)
    scores = (drugs @ targets.T) / temperature
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[positive_pairs[:, 0], positive_pairs[:, 1]] = True
    def direction(values, positives):
        active = positives.any(dim=1)
        values, positives = values[active], positives[active]
        numerator = torch.logsumexp(values.masked_fill(~positives, -torch.inf), dim=1)
        return (torch.logsumexp(values, dim=1) - numerator).mean()
    return (direction(scores, mask) + direction(scores.T, mask.T)) * 0.5

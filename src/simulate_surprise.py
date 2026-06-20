#!/usr/bin/env python3
"""Monte Carlo simulation for the FIFA World Cup 2026 with Surprise & Upset Factor.

This script extends the baseline Hybrid GNN + BNN model to introduce a mathematically
rigorous 'Surprise/Upset Factor' for underdog teams (e.g., Morocco) without hard-coding outcomes.

Mathematical Mechanisms:
------------------------
1. Epistemic Uncertainty Injection:
   Standard BNN outputs logits by sampling weights from a variational posterior:
       theta^(s) ~ q(theta) = N(mu, sigma^2)
   For matches involving high-volatility/underdog teams, we scale up the posterior standard
   deviation (epistemic uncertainty) of the weights during the forward pass:
       sigma_scaled = sigma * (1 + surprise_coeff * team_volatility)
   This increases the variance of weight samples theta^(s) across Monte Carlo draws.

2. Momentum-Weighted Prior (Logit Dispersion):
   To account for a team's momentum/form factor, we add a stochastic noise term to the predicted
   logits during the forward pass. This increases the spread of the predicted logits:
       z^(s)_final = z^(s)_base + delta^(s)
       delta^(s) ~ N(0, (momentum_coeff * team_momentum)^2 * I)
   This represents a dynamic, form-dependent prior over the logit outcomes.

3. Impact on Softmax Entropy:
   By increasing the sample-to-sample logit variance through both epistemic scaling and logit dispersion,
   the individual Monte Carlo samples of logits z^(s) fluctuate more widely. When passed through
   Softmax and averaged over S evaluations:
       p_final = (1 / S) * sum_{s=1}^S Softmax(z^(s))
   the resulting averaged probability distribution becomes flatter (higher entropy, moving closer
   to uniform [1/3, 1/3, 1/3]). This naturally increases the probability of "Black Swan" upsets in the
   Monte Carlo simulations without shifting the mean prediction mu.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run the simulation. Please run in an environment with PyTorch installed."
    ) from exc

TEAM_CODE_COLUMNS = ("home_team_code", "away_team_code")
CATEGORICAL_COLUMNS = ("tournament_code", "country_code")


@dataclass
class TrainConfig:
    data_path: Path = Path("prepared_world_cup_training_data.csv")
    output_dir: Path = Path("gnn_bnn_artifacts")
    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    node_hidden_dim: int = 96
    node_embedding_dim: int = 64
    bayes_hidden_dim: int = 192
    categorical_embedding_dim: int = 16
    dropout: float = 0.15
    knn_edges: int = 8
    mc_train_samples: int = 2
    mc_eval_samples: int = 30
    kl_beta: float = 1e-4
    log_loss_alpha: float = 0.70
    confidence_penalty: float = 0.50
    wrong_confidence_power: float = 2.0
    gradient_clip_norm: float = 2.0
    prior_sigma: float = 1.0
    seed: int = 42
    num_workers: int = 0
    use_amp: bool = True
    device: str = "auto"
    require_gpu: bool = False


# Configure surprise and momentum coefficients
SURPRISE_CONFIG = {
    # Volatility factor (scales posterior weight variance - epistemic uncertainty)
    "surprise_coeff": 2.2,     # Fine-tuned up from 1.5 to make upsets more pronounced
    # Momentum factor (scales logit dispersion prior)
    "momentum_coeff": 1.1,     # Fine-tuned up from 0.8 to widen logit spread stochastically
    # Team-specific factors (0.0 means default baseline model behavior)
    "team_volatility": {
        "Morocco": 0.85,        # High volatility / giant-killer potential
        "South Korea": 0.50,
        "Japan": 0.40,
        "Saudi Arabia": 0.60,
        "Canada": 0.45,
        "United States": 0.40,
        "Spain": 0.75,         # Fine-tune volatility to allow for high-level variance and upsets
        "Argentina": 0.75,     # Fine-tune volatility to allow for high-level variance and upsets
        "France": 0.50,        # Moderate volatility
        "Brazil": 0.55,        # Moderate volatility
    },
    "team_momentum": {
        "Morocco": 0.90,        # High momentum/recent form multiplier
        "Japan": 0.70,
        "Colombia": 0.60,
        "Spain": 0.50,
        "Argentina": 0.50,
    }
}


def chronological_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave a non-empty test split.")
    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def build_node_features(train_df: pd.DataFrame, continuous_columns: Sequence[str], num_nodes: int) -> np.ndarray:
    sums = np.zeros((num_nodes, len(continuous_columns)), dtype=np.float64)
    counts = np.zeros(num_nodes, dtype=np.float64)
    values = train_df.loc[:, continuous_columns].to_numpy(dtype=np.float32, copy=False)
    home_codes = train_df["home_team_code"].to_numpy(dtype=np.int64, copy=False)
    away_codes = train_df["away_team_code"].to_numpy(dtype=np.int64, copy=False)
    np.add.at(sums, home_codes, values)
    np.add.at(sums, away_codes, values)
    np.add.at(counts, home_codes, 1.0)
    np.add.at(counts, away_codes, 1.0)
    counts = np.maximum(counts, 1.0)
    node_features = sums / counts[:, None]
    active = counts > 1.0
    mean = node_features[active].mean(axis=0) if active.any() else np.zeros(len(continuous_columns))
    std = node_features[active].std(axis=0) if active.any() else np.ones(len(continuous_columns))
    std[std == 0.0] = 1.0
    return ((node_features - mean) / std).astype(np.float32)


def build_normalized_adjacency(
    train_df: pd.DataFrame,
    node_features: np.ndarray,
    num_nodes: int,
    knn_edges: int,
    device: torch.device,
) -> torch.Tensor:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    home = train_df["home_team_code"].to_numpy(dtype=np.int64, copy=False)
    away = train_df["away_team_code"].to_numpy(dtype=np.int64, copy=False)
    weights = train_df["sample_weight"].to_numpy(dtype=np.float32, copy=False)
    np.add.at(adjacency, (home, away), weights)
    np.add.at(adjacency, (away, home), weights)

    if knn_edges > 0 and num_nodes > 1:
        norms = np.linalg.norm(node_features, axis=1, keepdims=True)
        normalized = node_features / np.maximum(norms, 1e-8)
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        k = min(knn_edges, num_nodes - 1)
        neighbors = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        for source in range(num_nodes):
            for target in neighbors[source]:
                sim = similarity[source, target]
                if np.isfinite(sim) and sim > 0.0:
                    adjacency[source, target] += 0.25 * float(sim)
                    adjacency[target, source] += 0.25 * float(sim)

    adjacency += np.eye(num_nodes, dtype=np.float32)
    degrees = adjacency.sum(axis=1)
    inv_sqrt_degrees = np.power(np.maximum(degrees, 1e-8), -0.5)
    adjacency = inv_sqrt_degrees[:, None] * adjacency * inv_sqrt_degrees[None, :]
    row, col = np.nonzero(adjacency)
    values = adjacency[row, col]
    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values_tensor, (num_nodes, num_nodes), device=device).coalesce()


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.sparse.mm(adjacency, node_features))


class GraphEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GraphConvolution(in_features, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.conv1(node_features, adjacency))
        hidden = self.dropout(hidden)
        return self.conv2(hidden, adjacency)


class BayesianLinear(nn.Module):
    """Bayesian linear layer with mean-field Gaussian posterior and custom epistemic scaling."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0.0, 0.05))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).normal_(-5.0, 0.05))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features).normal_(-5.0, 0.05))

    @staticmethod
    def sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def forward(self, inputs: torch.Tensor, sample: bool, epistemic_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with option to scale weight variance dynamically per-sample."""
        weight_sigma = self.sigma(self.weight_rho)
        bias_sigma = self.sigma(self.bias_rho)

        if sample:
            batch_size = inputs.shape[0]
            if epistemic_scale is not None:
                # Epistemic Uncertainty Injection: Scale posterior variance per sample.
                # epistemic_scale has shape (batch_size, 1).
                # Samples weights uniquely per row using batched matrix multiplication.
                eps_w = torch.randn(batch_size, self.out_features, self.in_features, device=inputs.device)
                eps_b = torch.randn(batch_size, self.out_features, device=inputs.device)
                
                w_mu = self.weight_mu.unsqueeze(0)  # (1, out, in)
                # Scale standard deviation by epistemic_scale:
                w_sigma = weight_sigma.unsqueeze(0) * epistemic_scale.unsqueeze(-1)  # (batch_size, out, in)
                weight = w_mu + w_sigma * eps_w  # (batch_size, out, in)
                
                b_mu = self.bias_mu.unsqueeze(0)  # (1, out)
                b_sigma = bias_sigma.unsqueeze(0) * epistemic_scale  # (batch_size, out)
                bias = b_mu + b_sigma * eps_b  # (batch_size, out)
                
                # batched gemm: weight @ inputs.unsqueeze(-1)
                outputs = torch.bmm(weight, inputs.unsqueeze(-1)).squeeze(-1) + bias
            else:
                # Standard common batch weight sample
                weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
                outputs = F.linear(inputs, weight, bias)
        else:
            outputs = F.linear(inputs, self.weight_mu, self.bias_mu)

        return outputs


class HybridGNNBNN(nn.Module):
    """Hybrid GNN + BNN Model with surprise/upset factor injection support."""

    def __init__(
        self,
        node_feature_dim: int,
        continuous_dim: int,
        num_tournaments: int,
        num_countries: int,
        config: TrainConfig,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            node_feature_dim,
            config.node_hidden_dim,
            config.node_embedding_dim,
            config.dropout,
        )
        self.tournament_embedding = nn.Embedding(num_tournaments, config.categorical_embedding_dim)
        self.country_embedding = nn.Embedding(num_countries, config.categorical_embedding_dim)
        pair_dim = config.node_embedding_dim * 4
        input_dim = continuous_dim + pair_dim + 2 * config.categorical_embedding_dim
        self.norm = nn.LayerNorm(input_dim)
        self.bayes1 = BayesianLinear(input_dim, config.bayes_hidden_dim, prior_sigma=config.prior_sigma)
        self.bayes2 = BayesianLinear(config.bayes_hidden_dim, config.bayes_hidden_dim // 2, prior_sigma=config.prior_sigma)
        self.output = BayesianLinear(config.bayes_hidden_dim // 2, 3, prior_sigma=config.prior_sigma)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        continuous: torch.Tensor,
        home: torch.Tensor,
        away: torch.Tensor,
        tournament: torch.Tensor,
        country: torch.Tensor,
        sample: bool = True,
        epistemic_scale: torch.Tensor | None = None,
        logit_dispersion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass propagating inputs through GNN and BNN with uncertainty features."""
        node_embeddings = self.graph_encoder(node_features, adjacency)
        home_embedding = node_embeddings[home]
        away_embedding = node_embeddings[away]
        pair_features = torch.cat(
            [
                home_embedding,
                away_embedding,
                home_embedding - away_embedding,
                home_embedding * away_embedding,
            ],
            dim=-1,
        )
        dense = torch.cat(
            [
                continuous,
                pair_features,
                self.tournament_embedding(tournament),
                self.country_embedding(country),
            ],
            dim=-1,
        )
        hidden = self.norm(dense)
        
        # Inject epistemic scaling into the BNN layers
        hidden = self.dropout(F.gelu(self.bayes1(hidden, sample=sample, epistemic_scale=epistemic_scale)))
        hidden = self.dropout(F.gelu(self.bayes2(hidden, sample=sample, epistemic_scale=epistemic_scale)))
        logits = self.output(hidden, sample=sample, epistemic_scale=epistemic_scale)

        if sample and logit_dispersion is not None:
            # Momentum-Weighted Prior (Logit Dispersion): Add dynamic stochastic noise.
            # logit_dispersion has shape (batch_size, 1).
            noise = torch.randn_like(logits) * logit_dispersion
            logits = logits + noise

        return logits


# Simulation configuration
SIMULATION_CONFIG: Dict[str, Any] = {
    "artifact_dir": Path("models"),
    "model_state_path": Path("models/hybrid_gnn_bnn_state.pt"),
    "metadata_path": Path("models/training_metadata.json"),
    "training_data_path": Path("bin/data/processed/prepared_world_cup_training_data.csv"),
    "category_map_path": Path("bin/data/processed/category_encoding_maps.json"),
    "output_csv": Path("outputs/world_cup_2026_simulation_results.csv"),
    "output_plot": Path("outputs/world_cup_2026_top20_win_probability.png"),
    "n_simulations": 10_000,
    "mc_samples_per_match": 30,
    "probability_batch_size": 512,
    "seed": 2026,
    "device": "auto",
    "default_host_country": "United States",
}

WORLD_CUP_2026_GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Iran", "New Zealand", "Belgium", "Egypt"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "USA": "United States",
    "United States of America": "United States",
}


@dataclass
class TeamProfile:
    elo: float
    matches_last_5: float
    win_rate_last_5: float
    avg_goals_for_last_5: float
    avg_goals_against_last_5: float
    avg_points_last_5: float
    avg_penalties_for_last_5: float
    avg_late_goals_for_last_5: float
    matches_last_10: float
    win_rate_last_10: float
    avg_goals_for_last_10: float
    avg_goals_against_last_10: float
    avg_points_last_10: float
    avg_penalties_for_last_10: float
    avg_late_goals_for_last_10: float


@dataclass
class Standing:
    team: str
    group: str
    points: int = 0
    gf: int = 0
    ga: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    placement: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def running_on_kaggle() -> bool:
    return Path("/kaggle/input").exists() and Path("/kaggle/working").exists()


def discover_file(filename: str, local_path: Path) -> Path:
    """Robust file discovery helper, scanning both local directories and Kaggle input folders."""
    if local_path.exists():
        return local_path

    candidate_paths = []
    if not local_path.is_absolute():
        candidate_paths.extend(
            [
                Path.cwd() / local_path,
                Path("/kaggle/working") / local_path,
                Path("/kaggle/working") / filename,
                Path("/kaggle/working") / "gnn_bnn_artifacts" / filename,
            ]
        )
    candidate_paths.append(local_path)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    search_roots = [Path("/kaggle/working"), Path("/kaggle/input"), Path.cwd(), Path("/kaggle")]
    for root in search_roots:
        if root.exists():
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]

    diagnostics = []
    for root in [Path("/kaggle/working"), Path("/kaggle/input"), Path.cwd()]:
        if root.exists():
            visible = sorted(str(path) for path in root.rglob("*") if path.is_file())[:80]
            diagnostics.append(f"{root}: {visible}")
        else:
            diagnostics.append(f"{root}: does not exist")
            
    raise FileNotFoundError(
        f"Could not find {filename}. Checked direct paths, /kaggle/working, /kaggle/input, /kaggle, "
        f"and {Path.cwd()}.\nVisible files:\n" + "\n".join(diagnostics)
    )


def discover_artifact_paths(
    local_artifact_dir: Path,
    explicit_state_path: Path | None = None,
    explicit_metadata_path: Path | None = None,
) -> Tuple[Path | None, Path]:
    metadata = None
    if explicit_metadata_path is not None and explicit_metadata_path.exists():
        metadata = explicit_metadata_path
    else:
        try:
            metadata = discover_file("training_metadata.json", local_artifact_dir / "training_metadata.json")
        except FileNotFoundError:
            metadata = None

    if explicit_state_path is not None and explicit_state_path.exists():
        state = explicit_state_path
    else:
        state = discover_file("hybrid_gnn_bnn_state.pt", local_artifact_dir / "hybrid_gnn_bnn_state.pt")
    return metadata, state


def resolve_output_path(path: Path) -> Path:
    if running_on_kaggle() and not path.is_absolute():
        return Path("/kaggle/working") / path
    return path


def normalize_team_name(team: str) -> str:
    return TEAM_ALIASES.get(team.strip(), team.strip())


def normalize_groups(groups: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    normalized = {group: [normalize_team_name(team) for team in teams] for group, teams in groups.items()}
    return normalized


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_category_maps(path: Path | None) -> Dict[str, Dict[str, int]]:
    # Fallback to defaults if map file isn't found
    FALLBACK_HOME_TEAM_CODES = {
        "Mexico": 178, "South Africa": 261, "South Korea": 262, "Czech Republic": 70,
        "Canada": 46, "Bosnia and Herzegovina": 35, "Qatar": 220, "Switzerland": 270,
        "Brazil": 37, "Morocco": 185, "Haiti": 121, "Scotland": 245, "United States": 296,
        "Paraguay": 212, "Australia": 16, "Turkey": 288, "Germany": 105, "Ivory Coast": 137,
        "Ecuador": 80, "Curaçao": 68, "Netherlands": 190, "Japan": 139, "Sweden": 269,
        "Tunisia": 287, "Iran": 130, "New Zealand": 192, "Belgium": 27, "Egypt": 81,
        "Spain": 265, "Cape Verde": 48, "Saudi Arabia": 243, "Uruguay": 298, "France": 96,
        "Senegal": 248, "Iraq": 131, "Norway": 201, "Argentina": 12, "Algeria": 4,
        "Austria": 17, "Jordan": 141, "Portugal": 217, "DR Congo": 72, "Uzbekistan": 299,
        "Colombia": 59, "England": 85, "Croatia": 66, "Ghana": 106, "Panama": 209,
    }
    FALLBACK_AWAY_TEAM_CODES = {
        "Mexico": 176, "South Africa": 253, "South Korea": 254, "Czech Republic": 71,
        "Canada": 47, "Bosnia and Herzegovina": 36, "Qatar": 217, "Switzerland": 263,
        "Brazil": 38, "Morocco": 183, "Haiti": 120, "Scotland": 239, "United States": 290,
        "Paraguay": 209, "Australia": 17, "Turkey": 281, "Germany": 105, "Ivory Coast": 136,
        "Ecuador": 81, "Curaçao": 69, "Netherlands": 188, "Japan": 138, "Sweden": 262,
        "Tunisia": 280, "Iran": 129, "New Zealand": 190, "Belgium": 28, "Egypt": 82,
        "Spain": 257, "Cape Verde": 48, "Saudi Arabia": 238, "Uruguay": 292, "France": 96,
        "Senegal": 241, "Iraq": 130, "Norway": 199, "Argentina": 13, "Algeria": 4,
        "Austria": 18, "Jordan": 140, "Portugal": 214, "DR Congo": 73, "Uzbekistan": 293,
        "Colombia": 60, "England": 85, "Croatia": 67, "Ghana": 106, "Panama": 206,
    }
    FALLBACK_CATEGORY_MAPS = {
        "home_team": FALLBACK_HOME_TEAM_CODES,
        "away_team": FALLBACK_AWAY_TEAM_CODES,
        "tournament": {"FIFA World Cup": 61},
        "country": {"United States": 218},
    }
    
    if path is not None and path.exists():
        return load_json(path)
    return FALLBACK_CATEGORY_MAPS


def infer_train_config_from_state(state: Dict[str, torch.Tensor]) -> TrainConfig:
    config = TrainConfig()
    config.node_hidden_dim = int(state["graph_encoder.conv1.linear.weight"].shape[0])
    config.node_embedding_dim = int(state["graph_encoder.conv2.linear.weight"].shape[0])
    config.categorical_embedding_dim = int(state["tournament_embedding.weight"].shape[1])
    config.bayes_hidden_dim = int(state["bayes1.weight_mu"].shape[0])
    return config


def infer_continuous_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"target", "sample_weight", *TEAM_CODE_COLUMNS, *CATEGORICAL_COLUMNS, "class_id"}
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def build_scaler_from_training_data(df: pd.DataFrame, columns: Sequence[str], config: TrainConfig) -> Dict[str, Any]:
    train_df, _, _ = chronological_split(df, config.train_ratio, config.val_ratio)
    mean = train_df.loc[:, columns].mean(axis=0).astype("float32")
    std = train_df.loc[:, columns].std(axis=0).replace(0.0, 1.0).fillna(1.0).astype("float32")
    return {"columns": list(columns), "mean": mean.tolist(), "std": std.tolist()}


def build_metadata_fallback(df: pd.DataFrame, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    print("Metadata fallback triggered: Reconstructing columns and scaler values from CSV and model weights.")
    config = infer_train_config_from_state(state)
    continuous_columns = infer_continuous_columns(df)
    scaler = build_scaler_from_training_data(df, continuous_columns, config)
    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
        "continuous_columns": continuous_columns,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "team_code_columns": list(TEAM_CODE_COLUMNS),
        "scaler": scaler,
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize(values: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return ((values.astype(np.float32) - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def build_team_profiles(df: pd.DataFrame, category_maps: Dict[str, Dict[str, int]]) -> Dict[str, TeamProfile]:
    inverse_home = {code: team for team, code in category_maps["home_team"].items()}
    inverse_away = {code: team for team, code in category_maps["away_team"].items()}
    profile_values: Dict[str, Dict[str, float]] = {}

    def extract(row: pd.Series, prefix: str) -> Dict[str, float]:
        values = {"elo": float(row[f"{prefix}_elo_pre"])}
        for column in (
            "matches_last_5", "win_rate_last_5", "avg_goals_for_last_5", "avg_goals_against_last_5",
            "avg_points_last_5", "avg_penalties_for_last_5", "avg_late_goals_for_last_5",
            "matches_last_10", "win_rate_last_10", "avg_goals_for_last_10", "avg_goals_against_last_10",
            "avg_points_last_10", "avg_penalties_for_last_10", "avg_late_goals_for_last_10",
        ):
            values[column] = float(row[f"{prefix}_{column}"])
        return values

    ordered = df.reset_index(drop=True)
    for _, row in ordered.iterrows():
        home_team = inverse_home.get(int(row["home_team_code"]))
        away_team = inverse_away.get(int(row["away_team_code"]))
        if home_team:
            profile_values[home_team] = extract(row, "home")
        if away_team:
            profile_values[away_team] = extract(row, "away")

    average = pd.DataFrame(profile_values.values()).mean(axis=0).to_dict()
    profiles = {}
    for team in set(category_maps["home_team"]) | set(category_maps["away_team"]):
        values = {**average, **profile_values.get(team, {})}
        profiles[team] = TeamProfile(**values)
    return profiles


def build_h2h_store(df: pd.DataFrame, category_maps: Dict[str, Dict[str, int]]) -> Dict[frozenset, Dict[str, Any]]:
    inverse_home = {code: team for team, code in category_maps["home_team"].items()}
    inverse_away = {code: team for team, code in category_maps["away_team"].items()}
    store: Dict[frozenset, Dict[str, Any]] = defaultdict(lambda: {"wins": Counter(), "draws": 0, "total": 0})

    for row in df.itertuples(index=False):
        home = inverse_home.get(int(row.home_team_code))
        away = inverse_away.get(int(row.away_team_code))
        if not home or not away:
            continue
        key = frozenset((home, away))
        store[key]["total"] += 1
        if int(row.target) == 1:
            store[key]["wins"][home] += 1
        elif int(row.target) == -1:
            store[key]["wins"][away] += 1
        else:
            store[key]["draws"] += 1
    return store


def make_feature_row(
    home: str,
    away: str,
    continuous_columns: Sequence[str],
    profiles: Dict[str, TeamProfile],
    h2h_store: Dict[frozenset, Dict[str, Any]],
) -> np.ndarray:
    home_profile = profiles[home]
    away_profile = profiles[away]
    h2h = h2h_store.get(frozenset((home, away)), {"wins": Counter(), "draws": 0, "total": 0})
    total = max(int(h2h["total"]), 0)

    raw = {column: 0.0 for column in continuous_columns}
    raw.update(
        {
            "neutral": 1.0, "is_world_cup": 1.0, "is_friendly": 0.0, "is_competitive": 1.0,
            "home_elo_pre": home_profile.elo, "away_elo_pre": away_profile.elo,
            "elo_diff_pre": home_profile.elo - away_profile.elo, "h2h_matches": float(total),
            "h2h_home_team_win_rate": float(h2h["wins"][home] / total) if total else 0.0,
            "h2h_away_team_win_rate": float(h2h["wins"][away] / total) if total else 0.0,
            "h2h_draw_rate": float(h2h["draws"] / total) if total else 0.0,
            "year": 2026.0, "month": 6.0, "day_of_week": 5.0,
        }
    )

    for field_name, value in home_profile.__dict__.items():
        if field_name != "elo":
            raw[f"home_{field_name}"] = float(value)
    for field_name, value in away_profile.__dict__.items():
        if field_name != "elo":
            raw[f"away_{field_name}"] = float(value)

    return np.asarray([raw[column] for column in continuous_columns], dtype=np.float32)


def instantiate_model(
    metadata: Dict[str, Any],
    train_df: pd.DataFrame,
    state: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[HybridGNNBNN, torch.Tensor, torch.Tensor]:
    train_config = infer_train_config_from_state(state)
    train_split, _, _ = chronological_split(train_df, train_config.train_ratio, train_config.val_ratio)
    continuous_columns = metadata["continuous_columns"]
    num_nodes = int(train_df.loc[:, list(TEAM_CODE_COLUMNS)].max().max()) + 1
    node_features_np = build_node_features(train_split, continuous_columns, num_nodes)
    node_features = torch.tensor(node_features_np, dtype=torch.float32, device=device)
    adjacency = build_normalized_adjacency(train_split, node_features_np, num_nodes, train_config.knn_edges, device)

    model = HybridGNNBNN(
        node_feature_dim=node_features.shape[1],
        continuous_dim=len(continuous_columns),
        num_tournaments=int(state["tournament_embedding.weight"].shape[0]),
        num_countries=int(state["country_embedding.weight"].shape[0]),
        config=train_config,
    ).to(device)
    
    # Load state dict (preserves parameter compatibility since model shapes are identical)
    model.load_state_dict(state)
    model.eval()
    return model, node_features, adjacency


def precompute_pair_probabilities(
    teams: Sequence[str],
    model: HybridGNNBNN,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    metadata: Dict[str, Any],
    category_maps: Dict[str, Dict[str, int]],
    profiles: Dict[str, TeamProfile],
    h2h_store: Dict[frozenset, Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    mc_samples: int,
    host_country: str,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Precomputes Bayesian match probabilities for all pairings, injecting surprise factors."""
    continuous_columns = metadata["continuous_columns"]
    scaler = metadata["scaler"]
    tournament_code = category_maps["tournament"].get("FIFA World Cup", 61)
    country_code = category_maps["country"].get(host_country, 218)

    pairs = [(home, away) for home in teams for away in teams if home != away]
    probability_cache: Dict[Tuple[str, str], np.ndarray] = {}

    # Extract surprise scaling factors
    s_coeff = float(SURPRISE_CONFIG["surprise_coeff"])
    m_coeff = float(SURPRISE_CONFIG["momentum_coeff"])
    v_map = SURPRISE_CONFIG["team_volatility"]
    m_map = SURPRISE_CONFIG["team_momentum"]

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            
            # Compute match-level uncertainty values
            epistemic_scales = []
            logit_dispersions = []
            for home, away in batch_pairs:
                # 1. Epistemic scale: Combine team volatility & dynamic skill disparity (Elo Gap)
                vol_home = v_map.get(home, 0.0)
                vol_away = v_map.get(away, 0.0)
                
                # Dynamic skill disparity (Elo Gap) uncertainty injector
                elo_home = profiles[home].elo
                elo_away = profiles[away].elo
                elo_diff = abs(elo_home - elo_away)
                # Disparity factor scales up to 1.0 for large skill gaps, injecting variance
                disparity_vol = min(1.0, max(0.0, elo_diff - 150.0) / 400.0)
                
                max_vol = max(vol_home, vol_away, 0.5 * disparity_vol)
                # Scale posterior standard deviation
                epistemic_scales.append(1.0 + s_coeff * max_vol)

                # 2. Logit dispersion
                mom_home = m_map.get(home, 0.0)
                mom_away = m_map.get(away, 0.0)
                disp = m_coeff * (mom_home + mom_away)
                logit_dispersions.append(disp)

            # Move tensors to device
            epistemic_tensor = torch.tensor(epistemic_scales, dtype=torch.float32, device=device).unsqueeze(1)
            dispersion_tensor = torch.tensor(logit_dispersions, dtype=torch.float32, device=device).unsqueeze(1)

            raw_x = np.vstack(
                [
                    make_feature_row(home, away, continuous_columns, profiles, h2h_store)
                    for home, away in batch_pairs
                ]
            )
            x = torch.tensor(standardize(raw_x, scaler), dtype=torch.float32, device=device)
            home_codes = torch.tensor(
                [category_maps["home_team"].get(home, 0) for home, _ in batch_pairs],
                dtype=torch.long,
                device=device,
            )
            away_codes = torch.tensor(
                [category_maps["away_team"].get(away, 0) for _, away in batch_pairs],
                dtype=torch.long,
                device=device,
            )
            tournament = torch.full((len(batch_pairs),), tournament_code, dtype=torch.long, device=device)
            country = torch.full((len(batch_pairs),), country_code, dtype=torch.long, device=device)

            prob_sum = torch.zeros((len(batch_pairs), 3), dtype=torch.float32, device=device)
            
            # Draw posterior Monte Carlo predictive samples
            for _ in range(mc_samples):
                logits = model(
                    node_features=node_features,
                    adjacency=adjacency,
                    continuous=x,
                    home=home_codes,
                    away=away_codes,
                    tournament=tournament,
                    country=country,
                    sample=True,
                    epistemic_scale=epistemic_tensor,
                    logit_dispersion=dispersion_tensor
                )
                prob_sum += F.softmax(logits, dim=-1)

            probs = (prob_sum / float(mc_samples)).cpu().numpy()
            for pair, prob in zip(batch_pairs, probs):
                probability_cache[pair] = prob / prob.sum()

    return probability_cache


def sample_score(home: str, away: str, outcome_class: int, profiles: Dict[str, TeamProfile], rng: np.random.Generator) -> Tuple[int, int]:
    elo_diff = profiles[home].elo - profiles[away].elo
    strength = float(np.clip(abs(elo_diff) / 350.0, 0.0, 1.0))
    margin_probs = np.asarray([0.68 - 0.18 * strength, 0.23, 0.07 + 0.12 * strength, 0.02 + 0.06 * strength])
    margin_probs = margin_probs / margin_probs.sum()
    margin = int(rng.choice([1, 2, 3, 4], p=margin_probs))
    loser_goals = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))

    if outcome_class == 2:  # home win
        return loser_goals + margin, loser_goals
    if outcome_class == 0:  # away win
        return loser_goals, loser_goals + margin

    draw_goals = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.45, 0.25, 0.05]))
    return draw_goals, draw_goals


def play_match(
    home: str,
    away: str,
    probability_cache: Dict[Tuple[str, str], np.ndarray],
    profiles: Dict[str, TeamProfile],
    rng: np.random.Generator,
    knockout: bool = False,
    played_matches: Dict[Tuple[str, str], Tuple[int, int]] | None = None,
) -> Tuple[str | None, int, int, int]:
    
    # 1. Load actual real-life scores from results.csv if match has already transpired
    if not knockout and played_matches and (home, away) in played_matches:
        home_goals, away_goals = played_matches[(home, away)]
        if home_goals > away_goals:
            outcome = 2
            winner = home
        elif home_goals < away_goals:
            outcome = 0
            winner = away
        else:
            outcome = 1
            winner = None
        return winner, home_goals, away_goals, outcome

    # 2. Otherwise run surprise-adjusted posterior sampling
    probs = probability_cache[(home, away)]
    outcome = int(rng.choice([0, 1, 2], p=probs))  # 0: away, 1: draw, 2: home
    home_goals, away_goals = sample_score(home, away, outcome, profiles, rng)

    if not knockout:
        winner = home if outcome == 2 else away if outcome == 0 else None
        return winner, home_goals, away_goals, outcome

    if outcome == 2:
        return home, home_goals, away_goals, outcome
    if outcome == 0:
        return away, home_goals, away_goals, outcome

    # Shootout (draw resolution) using Elo-weighted Bradley-Terry formulation
    p_home = 1.0 / (1.0 + 10.0 ** (-(profiles[home].elo - profiles[away].elo) / 400.0))
    winner = home if rng.random() < p_home else away
    return winner, home_goals, away_goals, outcome


def rank_standings(standings: Iterable[Standing], profiles: Dict[str, TeamProfile], rng: np.random.Generator) -> List[Standing]:
    ranked = sorted(
        standings,
        key=lambda row: (
            row.points,
            row.gd,
            row.gf,
            row.wins,
            profiles[row.team].elo,
            rng.random(),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row.placement = index
    return ranked


def simulate_group(
    group_name: str,
    teams: Sequence[str],
    probability_cache: Dict[Tuple[str, str], np.ndarray],
    profiles: Dict[str, TeamProfile],
    rng: np.random.Generator,
    played_matches: Dict[Tuple[str, str], Tuple[int, int]] | None = None,
) -> List[Standing]:
    standings = {team: Standing(team=team, group=group_name) for team in teams}
    for home, away in combinations(teams, 2):
        winner, home_goals, away_goals, _ = play_match(
            home=home,
            away=away,
            probability_cache=probability_cache,
            profiles=profiles,
            rng=rng,
            knockout=False,
            played_matches=played_matches
        )
        standings[home].gf += home_goals
        standings[home].ga += away_goals
        standings[away].gf += away_goals
        standings[away].ga += home_goals

        if winner == home:
            standings[home].points += 3
            standings[home].wins += 1
            standings[away].losses += 1
        elif winner == away:
            standings[away].points += 3
            standings[away].wins += 1
            standings[home].losses += 1
        else:
            standings[home].points += 1
            standings[away].points += 1
            standings[home].draws += 1
            standings[away].draws += 1

    return rank_standings(standings.values(), profiles, rng)


def select_qualifiers(group_results: Dict[str, List[Standing]], profiles: Dict[str, TeamProfile], rng: np.random.Generator) -> List[Standing]:
    qualifiers = []
    third_placed = []
    for ranked in group_results.values():
        qualifiers.extend(ranked[:2])
        third_placed.append(ranked[2])

    best_thirds = sorted(
        third_placed,
        key=lambda row: (
            row.points,
            row.gd,
            row.gf,
            row.wins,
            profiles[row.team].elo,
            rng.random(),
        ),
        reverse=True,
    )[:8]
    qualifiers.extend(best_thirds)
    return qualifiers


def seed_round_of_32(qualifiers: List[Standing], profiles: Dict[str, TeamProfile], rng: np.random.Generator) -> List[str]:
    ranked = sorted(
        qualifiers,
        key=lambda row: (
            -row.placement,
            row.points,
            row.gd,
            row.gf,
            profiles[row.team].elo,
            rng.random(),
        ),
        reverse=True,
    )
    seed_order = [0, 31, 15, 16, 7, 24, 8, 23, 3, 28, 12, 19, 4, 27, 11, 20, 1, 30, 14, 17, 6, 25, 9, 22, 2, 29, 13, 18, 5, 26, 10, 21]
    bracket = [ranked[index] for index in seed_order]

    # Resolve same-group first-round matches
    for index in range(0, len(bracket), 2):
        if bracket[index].group != bracket[index + 1].group:
            continue
        for swap_index in range(index + 2, len(bracket)):
            if bracket[index].group != bracket[swap_index].group:
                bracket[index + 1], bracket[swap_index] = bracket[swap_index], bracket[index + 1]
                break

    return [row.team for row in bracket]


def play_knockout_round(
    bracket: Sequence[str],
    probability_cache: Dict[Tuple[str, str], np.ndarray],
    profiles: Dict[str, TeamProfile],
    rng: np.random.Generator,
) -> List[str]:
    winners = []
    for index in range(0, len(bracket), 2):
        home, away = bracket[index], bracket[index + 1]
        winner, _, _, _ = play_match(home, away, probability_cache, profiles, rng, knockout=True)
        winners.append(winner)
    return winners


def simulate_tournament_once(
    groups: Dict[str, List[str]],
    probability_cache: Dict[Tuple[str, str], np.ndarray],
    profiles: Dict[str, TeamProfile],
    rng: np.random.Generator,
    played_matches: Dict[Tuple[str, str], Tuple[int, int]] | None = None,
) -> Dict[str, List[str] | str]:
    group_results = {
        group_name: simulate_group(group_name, teams, probability_cache, profiles, rng, played_matches)
        for group_name, teams in groups.items()
    }
    qualifiers = select_qualifiers(group_results, profiles, rng)
    r32 = [row.team for row in qualifiers]
    bracket = seed_round_of_32(qualifiers, profiles, rng)

    r16 = play_knockout_round(bracket, probability_cache, profiles, rng)
    qf = play_knockout_round(r16, probability_cache, profiles, rng)
    sf = play_knockout_round(qf, probability_cache, profiles, rng)
    final = play_knockout_round(sf, probability_cache, profiles, rng)
    champion = play_knockout_round(final, probability_cache, profiles, rng)[0]

    return {"R32": r32, "R16": r16, "QF": qf, "SF": sf, "Final": final, "Winner": champion}


def run_simulations(
    groups: Dict[str, List[str]],
    probability_cache: Dict[Tuple[str, str], np.ndarray],
    profiles: Dict[str, TeamProfile],
    n_simulations: int,
    seed: int,
    played_matches: Dict[Tuple[str, str], Tuple[int, int]] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [team for group_teams in groups.values() for team in group_teams]
    counters = {stage: Counter() for stage in ("R32", "R16", "QF", "SF", "Final", "Winner")}

    for simulation_index in range(1, n_simulations + 1):
        result = simulate_tournament_once(groups, probability_cache, profiles, rng, played_matches)
        for stage in ("R32", "R16", "QF", "SF", "Final"):
            counters[stage].update(result[stage])
        counters["Winner"].update([result["Winner"]])
        if simulation_index % max(n_simulations // 10, 1) == 0:
            print(f"Completed {simulation_index:,}/{n_simulations:,} simulations")

    rows = []
    for team in teams:
        rows.append(
            {
                "Team": team,
                "Reach R32 %": counters["R32"][team] / n_simulations * 100.0,
                "Reach R16 %": counters["R16"][team] / n_simulations * 100.0,
                "Reach QF %": counters["QF"][team] / n_simulations * 100.0,
                "Reach SF %": counters["SF"][team] / n_simulations * 100.0,
                "Reach Final %": counters["Final"][team] / n_simulations * 100.0,
                "Win Tournament %": counters["Winner"][team] / n_simulations * 100.0,
            }
        )

    return pd.DataFrame(rows).sort_values("Win Tournament %", ascending=False).reset_index(drop=True)


def main() -> None:
    config = SIMULATION_CONFIG.copy()
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    # Locate required assets with robust fallback/search roots
    metadata_path, state_path = discover_artifact_paths(
        Path(config["artifact_dir"]),
        explicit_state_path=Path(config["model_state_path"]) if config.get("model_state_path") else None,
        explicit_metadata_path=Path(config["metadata_path"]) if config.get("metadata_path") else None,
    )
    
    try:
        category_map_path = discover_file("category_encoding_maps.json", Path(config["category_map_path"]))
    except FileNotFoundError:
        category_map_path = None
        
    training_data_path = discover_file("prepared_world_cup_training_data.csv", Path(config["training_data_path"]))
    output_csv = resolve_output_path(Path(config["output_csv"]))

    category_maps = load_category_maps(category_map_path)
    groups = normalize_groups(WORLD_CUP_2026_GROUPS)

    device = resolve_device(str(config["device"]))
    state = torch.load(state_path, map_location=device)
    df = pd.read_csv(training_data_path, low_memory=False, memory_map=True)
    
    # Load metadata file if available; otherwise, automatically reconstruct it
    metadata = load_json(metadata_path) if metadata_path is not None else build_metadata_fallback(df, state)

    # Preload already-played group matches from results.csv (e.g. Spain 0-0 Cape Verde, Morocco 1-1 Brazil)
    played_matches = {}
    try:
        raw_results_path = discover_file("results.csv", Path("bin/data/raw/football_data/results.csv"))
        raw_results = pd.read_csv(raw_results_path)
        raw_results['date'] = pd.to_datetime(raw_results['date'])
        
        # Filter 2026 FIFA World Cup matches with scores
        wc_2026 = raw_results[(raw_results['tournament'] == 'FIFA World Cup') & (raw_results['date'].dt.year == 2026)]
        wc_played = wc_2026.dropna(subset=['home_score', 'away_score'])
        
        for row in wc_played.itertuples():
            home = normalize_team_name(row.home_team)
            away = normalize_team_name(row.away_team)
            played_matches[(home, away)] = (int(row.home_score), int(row.away_score))
            played_matches[(away, home)] = (int(row.away_score), int(row.home_score))
            
        print(f"Loaded {len(wc_played)} actual played World Cup 2026 match results from results.csv.")
    except Exception as e:
        print(f"Could not load played matches from results.csv (using model projections only). Detail: {e}")

    print(f"Device: {device}")
    print(f"Model weights loaded from: {state_path}")
    print(f"Metadata status: {'Loaded' if metadata_path else 'Dynamically Reconstructed'}")
    print(f"Simulations Count: {int(config['n_simulations']):,}")
    print("Underdog Volatility Scaling enabled for: ", list(SURPRISE_CONFIG["team_volatility"].keys()))

    profiles = build_team_profiles(df, category_maps)
    h2h_store = build_h2h_store(df, category_maps)
    model, node_features, adjacency = instantiate_model(metadata, df, state, device)

    teams = [team for group_teams in groups.values() for team in group_teams]
    print("\nPrecomputing surprise-adjusted match probabilities...")
    probability_cache = precompute_pair_probabilities(
        teams=teams,
        model=model,
        node_features=node_features,
        adjacency=adjacency,
        metadata=metadata,
        category_maps=category_maps,
        profiles=profiles,
        h2h_store=h2h_store,
        device=device,
        batch_size=int(config["probability_batch_size"]),
        mc_samples=int(config["mc_samples_per_match"]),
        host_country=str(config["default_host_country"]),
    )

    # Let's inspect a sample matchup to verify the surprise factor
    print("\n--- Model Verification (Sample Matchups with Morocco) ---")
    morocco_vs_brazil = ("Morocco", "Brazil")
    morocco_vs_brazil_reverse = ("Brazil", "Morocco")
    
    if morocco_vs_brazil in probability_cache:
        # Prob classes: Away Win, Draw, Home Win
        p_m_vs_b = probability_cache[morocco_vs_brazil]
        print(f"Morocco (Home) vs Brazil (Away) Probs: Morocco Win: {p_m_vs_b[2]:.2%}, Draw: {p_m_vs_b[1]:.2%}, Brazil Win: {p_m_vs_b[0]:.2%}")
        
    if morocco_vs_brazil_reverse in probability_cache:
        p_b_vs_m = probability_cache[morocco_vs_brazil_reverse]
        print(f"Brazil (Home) vs Morocco (Away) Probs: Brazil Win: {p_b_vs_m[2]:.2%}, Draw: {p_b_vs_m[1]:.2%}, Morocco Win: {p_b_vs_m[0]:.2%}")

    print("\nRunning Monte Carlo simulations...")
    results = run_simulations(
        groups=groups,
        probability_cache=probability_cache,
        profiles=profiles,
        n_simulations=int(config["n_simulations"]),
        seed=int(config["seed"]),
        played_matches=played_matches
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print("\nTop 15 favorites with surprise scaling enabled:")
    print(results.head(15).to_string(index=False, float_format=lambda value: f"{value:6.2f}"))
    print(f"\nSaved simulation table to {output_csv}")


if __name__ == "__main__":
    main()

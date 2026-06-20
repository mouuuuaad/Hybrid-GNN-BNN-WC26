#!/usr/bin/env python3
"""Train a hybrid GNN + Bayesian Neural Network for match outcome prediction.

This script treats teams as graph nodes, propagates information over historical
team-vs-team topology with a sparse GCN, then uses Bayesian linear layers to
produce a posterior predictive distribution over match outcomes:

    class 0 = away win, class 1 = draw, class 2 = home win

The Bayesian layers use a diagonal Gaussian variational posterior q(theta) and
the reparameterization trick theta = mu + softplus(rho) * epsilon. Training
minimizes a variational objective:

    E_q[weighted_log_loss + ordinal_CRPS] + beta * KL(q(theta) || p(theta))

The KL term is the tractable part of the evidence lower bound (ELBO). The
Monte Carlo forward passes approximate the otherwise intractable posterior
predictive integral p(y|x,D) = integral p(y|x,theta) p(theta|D) dtheta.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    no_grad_decorator = torch.no_grad
except ModuleNotFoundError:  # Allows the file to be inspected/compiled without torch installed.
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    no_grad_decorator = lambda: (lambda function: function)


TARGET_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_TARGET = {0: -1, 1: 0, 2: 1}
TEAM_CODE_COLUMNS = ("home_team_code", "away_team_code")
CATEGORICAL_COLUMNS = ("tournament_code", "country_code")
NON_FEATURE_COLUMNS = ("target", "sample_weight", *TEAM_CODE_COLUMNS, *CATEGORICAL_COLUMNS)

# Notebook-first execution -------------------------------------------------
# Edit these values directly when running this whole file in Kaggle/Colab.
# If RUN_WITH_INLINE_CONFIG is True and the script detects a notebook kernel,
# argparse is skipped completely, so Colab's injected "-f kernel.json" cannot
# break execution.
RUN_WITH_INLINE_CONFIG = True
INLINE_CONFIG: Dict[str, Any] = {
    "data_path": Path("data/prepared_world_cup_training_data.csv"),
    "output_dir": Path("artifacts/model"),
    "epochs": 80,
    "batch_size": 1024,
    "learning_rate": 2e-3,
    "weight_decay": 1e-5,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "node_hidden_dim": 96,
    "node_embedding_dim": 64,
    "bayes_hidden_dim": 192,
    "categorical_embedding_dim": 16,
    "dropout": 0.15,
    "knn_edges": 8,
    "mc_train_samples": 2,
    "mc_eval_samples": 30,
    "kl_beta": 1e-4,
    "log_loss_alpha": 0.70,
    "confidence_penalty": 0.50,
    "wrong_confidence_power": 2.0,
    "gradient_clip_norm": 2.0,
    "prior_sigma": 1.0,
    "seed": 42,
    "num_workers": 0,
    "use_amp": True,
    "device": "auto",
    "require_gpu": False,
}


@dataclass
class TrainConfig:
    data_path: Path = Path("data/prepared_world_cup_training_data.csv")
    output_dir: Path = Path("artifacts/model")
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


def is_notebook_runtime() -> bool:
    return (
        "ipykernel" in sys.modules
        or "google.colab" in sys.modules
        or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    )


def config_from_inline() -> TrainConfig:
    valid_keys = set(TrainConfig.__dataclass_fields__)
    unknown_keys = sorted(set(INLINE_CONFIG) - valid_keys)
    if unknown_keys:
        raise ValueError(f"INLINE_CONFIG contains unknown keys: {unknown_keys}")
    return TrainConfig(**INLINE_CONFIG)


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a sparse-GCN + Bayesian MLP model.")
    parser.add_argument("--data-path", type=Path, default=TrainConfig.data_path)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--train-ratio", type=float, default=TrainConfig.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    parser.add_argument("--node-hidden-dim", type=int, default=TrainConfig.node_hidden_dim)
    parser.add_argument("--node-embedding-dim", type=int, default=TrainConfig.node_embedding_dim)
    parser.add_argument("--bayes-hidden-dim", type=int, default=TrainConfig.bayes_hidden_dim)
    parser.add_argument("--categorical-embedding-dim", type=int, default=TrainConfig.categorical_embedding_dim)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--knn-edges", type=int, default=TrainConfig.knn_edges)
    parser.add_argument("--mc-train-samples", type=int, default=TrainConfig.mc_train_samples)
    parser.add_argument("--mc-eval-samples", type=int, default=TrainConfig.mc_eval_samples)
    parser.add_argument("--kl-beta", type=float, default=TrainConfig.kl_beta)
    parser.add_argument("--log-loss-alpha", type=float, default=TrainConfig.log_loss_alpha)
    parser.add_argument("--confidence-penalty", type=float, default=TrainConfig.confidence_penalty)
    parser.add_argument("--wrong-confidence-power", type=float, default=TrainConfig.wrong_confidence_power)
    parser.add_argument("--gradient-clip-norm", type=float, default=TrainConfig.gradient_clip_norm)
    parser.add_argument("--prior-sigma", type=float, default=TrainConfig.prior_sigma)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--require-gpu", action="store_true", help="Stop immediately if CUDA is unavailable.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    args, unknown_args = parser.parse_known_args(argv)
    notebook_kernel_args = [arg for arg in unknown_args if arg == "-f" or "kernel-" in arg]
    unexpected_args = [arg for arg in unknown_args if arg not in notebook_kernel_args]
    if unexpected_args:
        parser.error(f"unrecognized arguments: {' '.join(unexpected_args)}")
    config = TrainConfig(**{key: value for key, value in vars(args).items() if key != "no_amp"})
    config.use_amp = not args.no_amp
    return config


def running_on_kaggle() -> bool:
    return Path("/kaggle/input").exists() and Path("/kaggle/working").exists()


def discover_kaggle_csv(filename: str = "prepared_world_cup_training_data.csv") -> Path | None:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    matches = sorted(kaggle_input.rglob(filename))
    return matches[0] if matches else None


def resolve_runtime_paths(config: TrainConfig) -> TrainConfig:
    """Make default paths work both locally and in Kaggle notebooks."""

    default_data_path = TrainConfig.data_path
    default_output_dir = TrainConfig.output_dir

    if config.data_path == default_data_path and not config.data_path.exists():
        kaggle_csv = discover_kaggle_csv(default_data_path.name)
        if kaggle_csv is not None:
            config.data_path = kaggle_csv

    if running_on_kaggle() and config.output_dir == default_output_dir:
        config.output_dir = Path("/kaggle/working") / default_output_dir

    return config


def require_torch() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is not installed. Install it first, for example:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "or use the CUDA wheel matching your GPU from https://pytorch.org/get-started/locally/"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_hardware_diagnostics(device: torch.device) -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    elif running_on_kaggle():
        print("Kaggle GPU is not attached to this session. Enable Notebook Settings -> Accelerator -> GPU, then restart the session.")


def autocast_context(device: torch.device, enabled: bool):
    """Use the modern torch.amp API while staying compatible with older wheels."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create a GradScaler without triggering PyTorch's deprecated CUDA path."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path, low_memory=False, memory_map=True)
    missing = {"target", *TEAM_CODE_COLUMNS, *CATEGORICAL_COLUMNS, "sample_weight"} - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["target", *TEAM_CODE_COLUMNS]).copy()
    df["class_id"] = df["target"].map(TARGET_TO_CLASS).astype("int64")
    sort_columns = [col for col in ("year", "month", "day_of_week") if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be valid fractions.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave a non-empty test split.")
    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def continuous_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_columns if col not in NON_FEATURE_COLUMNS and col != "class_id"]


def standardize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[float]]]:
    mean = train_df.loc[:, columns].mean(axis=0).astype("float32")
    std = train_df.loc[:, columns].std(axis=0).replace(0.0, 1.0).fillna(1.0).astype("float32")

    def transform(frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, columns].astype("float32")
        return ((values - mean) / std).to_numpy(dtype=np.float32, copy=True)

    scaler = {"columns": list(columns), "mean": mean.tolist(), "std": std.tolist()}
    return transform(train_df), transform(val_df), transform(test_df), scaler


def infer_num_nodes(*frames: pd.DataFrame) -> int:
    max_code = 0
    for frame in frames:
        max_code = max(max_code, int(frame.loc[:, list(TEAM_CODE_COLUMNS)].max().max()))
    return max_code + 1


def build_node_features(train_df: pd.DataFrame, continuous_columns: Sequence[str], num_nodes: int) -> np.ndarray:
    """Aggregate train-only team descriptors used by both GCN and kNN topology.

    The GCN node feature matrix X is not built from future rows. Each team gets
    the mean of match descriptors in rows where it appeared. This creates a
    stable empirical descriptor of team style/strength without leaking labels
    from validation/test rows into the graph encoder.
    """

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
    """Build A_hat = D^{-1/2}(A + I)D^{-1/2} as a sparse COO tensor.

    Historical matches provide observed edges. Cosine-similarity kNN edges add
    "shared feature" topology, allowing the graph to propagate signal between
    teams with similar empirical profiles even if they have limited direct H2H.
    """

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


class MatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, continuous_values: np.ndarray):
        self.x = torch.tensor(continuous_values, dtype=torch.float32)
        self.home = torch.tensor(frame["home_team_code"].to_numpy(np.int64), dtype=torch.long)
        self.away = torch.tensor(frame["away_team_code"].to_numpy(np.int64), dtype=torch.long)
        self.tournament = torch.tensor(frame["tournament_code"].to_numpy(np.int64), dtype=torch.long)
        self.country = torch.tensor(frame["country_code"].to_numpy(np.int64), dtype=torch.long)
        self.y = torch.tensor(frame["class_id"].to_numpy(np.int64), dtype=torch.long)
        self.sample_weight = torch.tensor(frame["sample_weight"].to_numpy(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.home[index],
            self.away[index],
            self.tournament[index],
            self.country[index],
            self.y[index],
            self.sample_weight[index],
        )


class GraphConvolution(nn.Module if nn is not None else object):
    """First-order spectral GCN layer: H' = sigma(A_hat H W + b)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.sparse.mm(adjacency, node_features)
        return self.linear(propagated)


class GraphEncoder(nn.Module if nn is not None else object):
    def __init__(self, in_features: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GraphConvolution(in_features, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.conv1(node_features, adjacency))
        hidden = self.dropout(hidden)
        return self.conv2(hidden, adjacency)


class BayesianLinear(nn.Module if nn is not None else object):
    """Bayesian affine layer with mean-field Gaussian variational posterior."""

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

    def forward(self, inputs: torch.Tensor, sample: bool) -> torch.Tensor:
        weight_sigma = self.sigma(self.weight_rho)
        bias_sigma = self.sigma(self.bias_rho)
        if sample:
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(inputs, weight, bias)

    def kl_loss(self) -> torch.Tensor:
        prior_var = self.prior_sigma**2

        def kl_gaussian(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return (
                torch.log(torch.tensor(self.prior_sigma, device=mu.device, dtype=mu.dtype) / sigma)
                + (sigma.pow(2) + mu.pow(2)) / (2.0 * prior_var)
                - 0.5
            ).sum()

        return kl_gaussian(self.weight_mu, self.sigma(self.weight_rho)) + kl_gaussian(
            self.bias_mu, self.sigma(self.bias_rho)
        )


class HybridGNNBNN(nn.Module if nn is not None else object):
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
    ) -> torch.Tensor:
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
        hidden = self.dropout(F.gelu(self.bayes1(hidden, sample=sample)))
        hidden = self.dropout(F.gelu(self.bayes2(hidden, sample=sample)))
        return self.output(hidden, sample=sample)

    def kl_loss(self) -> torch.Tensor:
        return self.bayes1.kl_loss() + self.bayes2.kl_loss() + self.output.kl_loss()


def ordinal_crps(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Discrete CRPS for ordered classes away win < draw < home win."""

    observed = F.one_hot(labels, num_classes=3).to(dtype=probs.dtype)
    pred_cdf = torch.cumsum(probs, dim=-1)
    obs_cdf = torch.cumsum(observed, dim=-1)
    return torch.mean((pred_cdf - obs_cdf).pow(2), dim=-1)


def variational_match_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    class_weight: torch.Tensor,
    alpha: float,
    confidence_penalty: float,
    wrong_confidence_power: float,
) -> torch.Tensor:
    """Hybrid differentiable scoring rule with explicit overconfidence control.

    The logarithmic term is strictly proper and punishes assigning low mass to
    the observed class. The CRPS term respects the ordinal geometry of outcomes:
    predicting home win for an away win is worse than predicting draw. The final
    wrong-class power term adds curvature against confidently wrong posterior
    samples, which is useful for tournament forecasting where tail risk matters.
    """

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
    label_weights = class_weight[labels]
    wrong_mask = 1.0 - F.one_hot(labels, num_classes=3).to(dtype=probs.dtype)
    wrong_confidence = torch.sum((probs * wrong_mask).pow(wrong_confidence_power), dim=-1)
    dynamic_nll = nll * (1.0 + confidence_penalty * wrong_confidence)
    crps = ordinal_crps(probs, labels)
    per_row = alpha * dynamic_nll + (1.0 - alpha) * crps
    return torch.mean(per_row * sample_weight * label_weights)


def move_batch(batch: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


def train_one_epoch(
    model: HybridGNNBNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    class_weight: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    scaler,
    use_amp: bool,
    train_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0

    for batch in loader:
        continuous, home, away, tournament, country, labels, sample_weight = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            data_loss = 0.0
            for _ in range(config.mc_train_samples):
                logits = model(node_features, adjacency, continuous, home, away, tournament, country, sample=True)
                data_loss = data_loss + variational_match_loss(
                    logits,
                    labels,
                    sample_weight,
                    class_weight,
                    config.log_loss_alpha,
                    config.confidence_penalty,
                    config.wrong_confidence_power,
                )
            data_loss = data_loss / float(config.mc_train_samples)
            loss = data_loss + config.kl_beta * model.kl_loss() / float(train_size)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        seen += batch_size

    return total_loss / max(seen, 1)


@no_grad_decorator()
def posterior_predictive(
    model: HybridGNNBNN,
    loader: DataLoader,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    class_weight: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    all_weights = []
    all_expected_entropy = []

    for batch in loader:
        continuous, home, away, tournament, country, labels, sample_weight = move_batch(batch, device)
        mc_probs = []
        for _ in range(config.mc_eval_samples):
            logits = model(node_features, adjacency, continuous, home, away, tournament, country, sample=True)
            mc_probs.append(F.softmax(logits, dim=-1))
        stacked = torch.stack(mc_probs, dim=0)
        mean_probs = stacked.mean(dim=0)
        expected_entropy = (-stacked * torch.log(stacked.clamp_min(1e-8))).sum(dim=-1).mean(dim=0)
        all_probs.append(mean_probs.cpu())
        all_labels.append(labels.cpu())
        all_weights.append(sample_weight.cpu())
        all_expected_entropy.append(expected_entropy.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    weights = torch.cat(all_weights)
    expected_entropy = torch.cat(all_expected_entropy)
    predictions = probs.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    weighted_accuracy = ((predictions == labels).float() * weights).sum().item() / weights.sum().item()
    nll = -torch.log(probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-8))
    crps = ordinal_crps(probs, labels)
    predictive_entropy = (-probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    epistemic_uncertainty = predictive_entropy - expected_entropy
    scoring_loss = variational_match_loss(
        torch.log(probs.clamp_min(1e-8)),
        labels,
        weights,
        class_weight.cpu(),
        config.log_loss_alpha,
        config.confidence_penalty,
        config.wrong_confidence_power,
    )
    return {
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "nll": nll.mean().item(),
        "weighted_nll": (nll * weights).sum().item() / weights.sum().item(),
        "crps": crps.mean().item(),
        "weighted_scoring_loss": float(scoring_loss),
        "predictive_entropy": predictive_entropy.mean().item(),
        "aleatoric_uncertainty": expected_entropy.mean().item(),
        "epistemic_uncertainty": epistemic_uncertainty.clamp_min(0.0).mean().item(),
    }


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(dataset: MatchDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def save_artifacts(
    output_dir: Path,
    model: HybridGNNBNN,
    config: TrainConfig,
    scaler: Dict[str, List[float]],
    metrics: Dict[str, Dict[str, float]],
    continuous_columns: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "hybrid_gnn_bnn_state.pt")
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "continuous_columns": list(continuous_columns),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "team_code_columns": list(TEAM_CODE_COLUMNS),
        "target_to_class": TARGET_TO_CLASS,
        "class_to_target": CLASS_TO_TARGET,
        "scaler": scaler,
        "metrics": metrics,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(config: TrainConfig | None = None) -> None:
    if config is None:
        if RUN_WITH_INLINE_CONFIG and is_notebook_runtime():
            config = config_from_inline()
        else:
            config = parse_args()
    require_torch()
    config = resolve_runtime_paths(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    if config.require_gpu and device.type != "cuda":
        raise RuntimeError(
            "GPU was required but CUDA is unavailable. In Kaggle, enable Settings -> Accelerator -> GPU and restart."
        )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    use_amp = bool(config.use_amp and device.type == "cuda")

    df = load_frame(config.data_path)
    train_df, val_df, test_df = chronological_split(df, config.train_ratio, config.val_ratio)
    continuous_columns = continuous_feature_columns(df)
    train_x, val_x, test_x, scaler = standardize_splits(train_df, val_df, test_df, continuous_columns)
    num_nodes = infer_num_nodes(train_df, val_df, test_df)
    node_features_np = build_node_features(train_df, continuous_columns, num_nodes)
    node_features = torch.tensor(node_features_np, dtype=torch.float32, device=device)
    adjacency = build_normalized_adjacency(train_df, node_features_np, num_nodes, config.knn_edges, device)

    train_dataset = MatchDataset(train_df, train_x)
    val_dataset = MatchDataset(val_df, val_x)
    test_dataset = MatchDataset(test_df, test_x)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.num_workers, device)
    val_loader = make_loader(val_dataset, config.batch_size, False, config.num_workers, device)
    test_loader = make_loader(test_dataset, config.batch_size, False, config.num_workers, device)

    num_tournaments = int(df["tournament_code"].max()) + 1
    num_countries = int(df["country_code"].max()) + 1
    model = HybridGNNBNN(
        node_feature_dim=node_features.shape[1],
        continuous_dim=len(continuous_columns),
        num_tournaments=num_tournaments,
        num_countries=num_countries,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler_amp = make_grad_scaler(device, use_amp)
    weights = class_weights(train_df["class_id"].to_numpy(np.int64), device)

    best_val = math.inf
    best_state = None
    history: List[Dict[str, float]] = []
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {device} | AMP: {use_amp}")
    print_hardware_diagnostics(device)
    if device.type == "cpu":
        print("CUDA is not available. On Kaggle, enable Settings -> Accelerator -> GPU for faster training.")
    print(f"Rows: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"Graph: nodes={num_nodes:,} sparse_edges={adjacency._nnz():,}")
    print(f"Continuous features={len(continuous_columns):,}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            node_features,
            adjacency,
            weights,
            config,
            device,
            scaler_amp,
            use_amp,
            len(train_dataset),
        )
        scheduler.step()
        val_metrics = posterior_predictive(model, val_loader, node_features, adjacency, weights, config, device)
        epoch_record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_record)
        val_score = val_metrics["weighted_scoring_loss"]
        if val_score < best_val:
            best_val = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_score={val_score:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_epi={val_metrics['epistemic_uncertainty']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = posterior_predictive(model, val_loader, node_features, adjacency, weights, config, device)
    test_metrics = posterior_predictive(model, test_loader, node_features, adjacency, weights, config, device)
    metrics = {"validation": val_metrics, "test": test_metrics, "history": history}
    save_artifacts(config.output_dir, model, config, scaler, metrics, continuous_columns)
    print(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
    print(f"Saved model and metadata to {config.output_dir}")


if __name__ == "__main__":
    main()

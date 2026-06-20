#!/usr/bin/env python3
"""Monte Carlo simulation for the FIFA World Cup 2026.

The simulator loads the trained hybrid GNN + BNN model, reconstructs the graph
inputs from the same training data used during model fitting, precomputes
Bayesian posterior match probabilities for every possible ordered team pairing,
then runs repeated full-tournament simulations.

Output columns are percentages:
    Reach R32, Reach R16, Reach QF, Reach SF, Reach Final, Win Tournament
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
        "PyTorch is required to load the trained GNN+BNN model. "
        "Run this script in the same Kaggle/Colab environment used for training."
    ) from exc

try:
    from .train import (
        CATEGORICAL_COLUMNS,
        TEAM_CODE_COLUMNS,
        HybridGNNBNN,
        TrainConfig,
        build_node_features,
        build_normalized_adjacency,
        chronological_split,
    )
except ModuleNotFoundError:
    from train import (
        CATEGORICAL_COLUMNS,
        TEAM_CODE_COLUMNS,
        HybridGNNBNN,
        TrainConfig,
        build_node_features,
        build_normalized_adjacency,
        chronological_split,
    )

# ... existing code unchanged until WORLD_CUP_2026_GROUPS and below ...

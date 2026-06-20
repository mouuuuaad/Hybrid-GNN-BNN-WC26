# FIFA World Cup 2026 Forecasting

A hybrid Graph Neural Network + Bayesian Neural Network pipeline for predicting FIFA World Cup 2026 outcomes using Monte Carlo tournament simulation.

## Directory Structure

```
.
├── src/worldcup/              # Python package
│   ├── __init__.py
│   ├── train.py               # GNN+BNN training
│   ├── prepare.py             # Data preparation & feature engineering
│   ├── simulate.py            # Tournament simulation
│   └── dashboard.py           # (coming) Dashboard data generation
├── data/
│   ├── raw/                   # Input CSVs (results, shootouts, goalscorers, former_names)
│   └── *.csv                  # Prepared training data, encodings
├── artifacts/
│   └── model/                 # Trained model weights & metadata
├── web/                       # Dashboard & frontend
│   └── index.html
├── bin/                       # CLI entry scripts
├── requirements.txt
├── README.md
└── Makefile
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the Kaggle international football dataset (results.csv, shootouts.csv, goalscorers.csv, former_names.csv) in `data/raw/`.

```bash
python -m src.worldcup.prepare
```

This generates:
- `data/prepared_world_cup_training_data.csv`
- `data/category_encoding_maps.json`

### 3. Train Model

```bash
python -m src.worldcup.train --epochs 80 --batch-size 512
```

Outputs to `artifacts/model/`:
- `hybrid_gnn_bnn_state.pt` — trained weights
- `training_metadata.json` — config, scaler, metrics

### 4. Run Simulation

```bash
python -m src.worldcup.simulate
```

Generates:
- `world_cup_2026_simulation_results.csv` — tournament probabilities
- `world_cup_2026_top20_win_probability.png` — visualization

### 5. Build Dashboard

```bash
python -m src.worldcup.dashboard
```

Creates `web/` assets for the interactive frontend.

## Architecture

### Data Flow

1. **Prepare** (`prepare.py`): Cleans CSVs → chronological form → Elo ratings → model-ready CSV
2. **Train** (`train.py`): Builds sparse team graph → GCN encoder + Bayesian MLP → posterior predictive
3. **Simulate** (`simulate.py`): Loads trained model → precomputes match probabilities → runs 10k Monte Carlo tournaments
4. **Dashboard** (`dashboard.py`): Transforms simulation results → group tables, bracket, charts → JSON/JS for web UI

### Model: Hybrid GNN + BNN

- **Graph Encoder (GCN)**: Teams as nodes, historical/feature-similarity edges → per-team embeddings
- **Bayesian MLP**: Embeddings + match context → posterior over {away win, draw, home win}
- **Monte Carlo**: 30 forward passes per match → robust uncertainty quantification
- **Loss**: Weighted NLL + ordinal CRPS + KL divergence (variational ELBO)

## CLI Reference

### Prepare

```bash
python -m src.worldcup.prepare \
  --data-dir data/raw \
  --output data/prepared_world_cup_training_data.csv \
  --encoding-map-output data/category_encoding_maps.json
```

### Train

```bash
python -m src.worldcup.train \
  --data-path data/prepared_world_cup_training_data.csv \
  --output-dir artifacts/model \
  --epochs 80 \
  --batch-size 512 \
  --device auto  # or "cuda", "cpu"
```

### Simulate

```bash
python -m src.worldcup.simulate
```

Default paths resolved to new structure; override with env or config edits.

### Dashboard

```bash
python -m src.worldcup.dashboard \
  --results-csv world_cup_2026_simulation_results.csv \
  --output-dir web
```

## Development

### Running Tests

```bash
# Smoke test: check imports
python -c "from src.worldcup import train, prepare, simulate; print('✓ Imports OK')"
```

### Adding New Modules

1. Create `src/worldcup/mymodule.py`
2. Add to `src/worldcup/__init__.py`
3. Import from anywhere: `from src.worldcup import mymodule`

## Notes

- **Kaggle Compatibility**: Scripts auto-detect `/kaggle/working` and `/kaggle/input` for path resolution.
- **GPU**: Uses PyTorch AMP by default on CUDA; disable with `--no-amp`.
- **Reproducibility**: Fixed seeds (train: 42, simulate: 2026) for all runs.

## License & Attribution

Dataset: [Kaggle International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-19722024)

Model: Hybrid GNN + Bayesian NN forecasting pipeline (2026)

See LICENSE file for details.

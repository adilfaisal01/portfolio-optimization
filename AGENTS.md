# portfolio-optimization — AGENTS.md

This project uses a **SQLite codebase index** stored in the repo itself. Query the index to understand the codebase before making changes — no need to keep the full codebase in context.

## Codebase Index

**Location:** `.codebase/codebase_index.db` (in-repo, versionable)
**Last indexed:** 2026-07-05

### How to query

```bash
# List all files with summaries
python3 ~/.hermes/scripts/codebase_indexer.py --list --repo portfolio-optimization

# Search for specific files or concepts
python3 ~/.hermes/scripts/codebase_indexer.py --query "paper_trading" --repo portfolio-optimization
python3 ~/.hermes/scripts/codebase_indexer.py --query "MPC" --repo portfolio-optimization
python3 ~/.hermes/scripts/codebase_indexer.py --query "VAR" --repo portfolio-optimization
python3 ~/.hermes/scripts/codebase_indexer.py --query "JEPA" --repo portfolio-optimization

# List vault project notes (lab notebook entries)
python3 ~/.hermes/scripts/codebase_indexer.py --vault --repo portfolio-optimization

# Direct SQLite queries for deeper analysis
python3 -c "import sqlite3; c=sqlite3.connect('.codebase/codebase_index.db'); [print(r) for r in c.execute('SELECT path, purpose FROM files WHERE ext=\".py\" ORDER BY path')]"
```

### Re-index after changes

```bash
python3 ~/.hermes/scripts/codebase_indexer.py .
```

## Project Overview

Two major subsystems:

1. **VAR-MPC Portfolio Management** — Dynamic portfolio management using Model Predictive Control with Vector Autoregression across 7 ETFs.
2. **JEPA Representation Learning** — Self-supervised learning for market regime detection and forecasting using a Joint-Embedding Predictive Architecture.

### Key Files

| File | Purpose |
|------|---------|
| `dynamics.py` | MPC Planner (OSQP solver) + Market Simulator |
| `VAR_setup.py` | VAR model training, forecasting, ADF stationarity checks |
| `file.py` | Main backtest loop (2020-2025) |
| `paper_trading_bot.py` | Live paper trading bot — runs daily via Alpaca |
| `cascading_mpc_v3.py` | SE701 daily MPC + intraday drift correction via PID |
| `backtest_covar.py` | Static vs per-step covariance backtest comparison |
| `unit_tests.py` | Pytest suite covering planner, simulator, edge cases |
| `investors.py` | Benchmark comparison vs ARKK, BRK-B, SPY, PSHZF |
| `min_variance.py` | Min-variance benchmark strategy |
| `jepa-training.py` | JEPA model training — configurable via dataclasses + env vars |
| `regime_probe.py` | Regime classifier probe: do JEPA embeddings encode market regime? |
| `src/models/encoder.py` | JEPA encoder (Transformer) |
| `src/models/predictor.py` | JEPA predictor (Transformer) |
| `src/models/decoder.py` | JEPA decoder for downstream tasks |
| `src/models/tokenizer.py` | Tokenizer for JEPA input |
| `src/data_loaders/data_loader.py` | Data loaders for JEPA training/evaluation |
| `individual_stocks/data_class_parquet.py` | `StockMarketJEPADataset` — parquet-based dataset with VIX conditioning |

### Architecture

```
── VAR-MPC ──
paper_trading_bot.py  →  VAR_setup.VARAnalysis (forecast)
                      →  dynamics.MPCPLanner (solve QP)
                      →  Alpaca API (execute trades)

── JEPA ──
jepa-training.py      →  src/models/encoder.py (encode visible patches)
                      →  src/models/predictor.py (predict masked patches)
                      →  src/data_loaders/data_loader.py (masked patch sampling)
                      →  individual_stocks/data_class_parquet.py (parquet I/O)

regime_probe.py       →  src/models/encoder.py (frozen embeddings)
                      →  sklearn classifier (probe regime)
```

### Key Parameters

**VAR-MPC:**
- 7 assets (AGG, GLD, SLV, SPY, VTI, VEA, VWO), 40% max weight per asset
- 30-day planning horizon, VAR refit every 21 days
- VIX-scaled transaction costs (ρ = 2 × VIX_t / VIX_normal)
- 0.5% drift threshold for rebalancing
- Risk aversion γ = 1, quadratic transaction cost penalty

**JEPA Training (`jepa-training.py`):**
- `JEPA_parameters` (configurable via `JEPA_*` env vars):
  - `mask_ratio`: 0.7 (fraction of patches masked)
  - `num_patches`: 20 (total patches per sample)
  - `vix_fairweather`: 20 (VIX threshold for conditioning)
  - `predictor_embed_dim`: 128
  - `encoder_embed_dim`: 64
  - `kernel_size`: 49
  - `dim_in_encoder`: 49
  - `num_layers_encoder`: 4, `num_layers_predictor`: 2
  - `nhead_encoder`: 8, `n_head_predictor`: 4
- `Training_configuration` (configurable via `TRAIN_*` env vars):
  - `batch_size`: 4, `lr`: 3e-4, `weight_decay`: 0
  - `ema_momentum`: 0.998, `num_epochs`: 3
  - `lambda_v`: 2 (variance), `lambda_cv`: 2.1 (covariance) — VICReg loss weights
- Loss: VICReg (variance + invariance + covariance regularization)
- Optimizer: AdamW, EMA target network update

## Vault Lab Notebooks

Project notes live in `lab-notes/daily/` in the repo. These contain experiment logs, architecture decisions, and findings. Query them with `--vault` flag.

## Workflow

1. **Understand** — Query the index for relevant files + vault notes
2. **Plan** — Describe the change and which files to modify
3. **Implement** — Use OpenCode or direct editing
4. **Verify** — Run tests or check the output
5. **Re-index** — `python3 ~/.hermes/scripts/codebase_indexer.py .`

# Portfolio Optimization — VAR-MPC + JEPA

**Two-phase project: Model Predictive Control meets Self-Supervised Learning for portfolio management.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![OSQP](https://img.shields.io/badge/Solver-OSQP-green.svg) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)

---

## Overview

This project explores two complementary approaches to dynamic portfolio management:

1. **Phase 1 — VAR-MPC (`var_mpc/`):** A model-based control approach using Vector Autoregression for return forecasting and Model Predictive Control for optimal rebalancing.
2. **Phase 2 — JEPA (`jepa/`):** A self-supervised learning approach using a Joint-Embedding Predictive Architecture to learn market representations, then using those representations as a **KV retrieval system** to inform the MPC cost function.

The project manages 7 ETFs spanning bonds, equities, precious metals, and international markets.

| Ticker | Description | Asset Class |
| --- | --- | --- |
| **AGG** | iShares Core U.S. Aggregate Bond ETF | Bonds |
| **SPY** | SPDR S&P 500 ETF | US Stocks (Large Cap) |
| **GLD** | SPDR Gold Shares | Precious Metals |
| **SLV** | iShares Silver Trust | Precious Metals |
| **VTI** | Vanguard Total Stock Market ETF | US Stocks (Total Market) |
| **VEA** | Vanguard FTSE Developed Markets ETF | International Stocks (Developed) |
| **VWO** | Vanguard FTSE Emerging Markets ETF | International Stocks (Emerging) |

---

## Phase 1: VAR-MPC (2020–2025 Backtest)

### Methodology

The portfolio state is defined as `x_k = [S_k, w_k]ᵀ` where `S_k` is log-wealth and `w_k` is the weight vector. A VAR model forecasts returns over a 30-day horizon, and an MPC planner solves a quadratic program (via OSQP) to find optimal rebalancing trades.

**Cost function:**
```
J = Σ ( w_kᵀ Σ w_k   +   u_kᵀ R u_k )   -   γ S_N
    └── risk ──┘     └─ transaction cost ─┘   └─ terminal wealth ─┘
```

Transaction costs are scaled by the CBOE Volatility Index (VIX): `ρ = 2 × (VIX_t / VIX_normal)`, increasing cost during volatile periods.

**Constraints:** No short selling, 40% max concentration per asset, fully invested.

### Key Results (2020–2025)

| Metric | Buy & Hold | MPC (40% max) |
| --- | --- | --- |
| **CAGR** | 12.85% | **14.77%** |
| **Sharpe Ratio** | 0.61 | **0.73** |
| **Max Drawdown** | 25.34% | 26.37% |
| **Calmar Ratio** | 0.51 | **0.56** |
| **Annual Volatility** | 14.83% | 16.53% |
| **Mean Daily Turnover** | – | **0.47%** |

### Comparison with Notable Investors

| Strategy | Sharpe | CAGR | Max Drawdown | Calmar |
| --- | --- | --- | --- | --- |
| **MPC (40%)** | **0.73** | 14.77% | 26.37% | **0.56** |
| ARKK (Cathie Wood) | 0.34 | 8.08% | 80.91% | 0.10 |
| BRK-B (Buffett) | 0.59 | 14.11% | 29.57% | 0.47 |
| SPY (S&P 500) | 0.63 | 14.97% | 33.72% | 0.44 |
| PSHZF (Pershing Square) | **0.79** | **23.38%** | 32.96% | **0.70** |

The MPC strategy delivers superior risk-adjusted returns — second-highest Sharpe and best Calmar ratio among all compared strategies, with drawdown lower than every benchmark except its own variants.

### Key Findings

1. **40% weight cap** delivered the best trade-off: CAGR of 14.77%, Sharpe of 0.73, Calmar of 0.56
2. **VIX-scaled transaction costs** kept mean daily turnover low at 0.47%, preventing cost drag
3. Outperformed buy-and-hold, ARKK, and Berkshire Hathaway on a risk-adjusted basis
4. Avoids speculative concentration — unlike ARKK's 80.91% drawdown during the 2022–2023 tech downturn

---

## Phase 2: JEPA Representation Learning + KV Retrieval

### Motivation

The VAR-MPC approach relies on point forecasts from a linear model. What if we could learn richer representations of market state — capturing regime changes, volatility regimes, and latent market factors — and use them to **retrieve similar historical regimes** to inform the MPC cost function?

### Architecture

The JEPA encoder produces 1280-dimensional embeddings (20 patches × 64 dims) from 49-feature market windows (11 ETFs × 3 metrics + 5 macro × 3 + VIX). These embeddings serve as **keys** in a KV retrieval system:

```
JEPA encoder → z_t (query)
     ↓
KV store: 163 historical (embedding, returns, covariance) pairs
     ↓
Euclidean top-K retrieval → sparse attention (τ-weighted)
     ↓
r_weighted (11,), S_weighted (11, 11) → MPC cost function
     ↓
cost = -w @ r + λ(z) * w @ S @ w + τ * ||w - w_prev||₁
```

**Key properties:**
- **No training required** — JEPA encoder is pre-trained, everything else is retrieval + geometry
- **Counter-cyclical risk** — λ(z) from distance to bull center adjusts risk aversion
- **Regime-conditional covariance** — Σ comes from historical analogs, not rolling windows
- **No lookahead** — encoder was trained on 2007-2019 only, test set is out-of-sample

### Embedding Database

**Script:** `jepa/embedding_db/cost_fcn.py`

163 windows spanning 2007-01-11 → 2019-12-20, each with:

| Field | Shape | Description |
| --- | --- | --- |
| `start_date` | string | First day of the 20-day window |
| `end_date` | string | Last day of the 20-day window |
| `embedding` | (1280,) | Flattened JEPA encoder output |
| `vix_avg` | scalar | Mean normalized VIX over the window |
| `mean_returns` | (11,) | Mean daily log returns for 11 ETFs |
| `covariances` | (11, 11) | Covariance via Parkinson vol proxy + correlation |

### KV Retrieval + Sparse Attention

**Script:** `jepa/embedding_db/query.py`

Two functions:
1. `query_similar()` — Euclidean distance top-K retrieval
2. `weighted_financial_metrics()` — temperature-controlled sparse attention with einsum

**Validation results:**

| Regime | Top Match | Returns | Variance |
|--------|-----------|---------|----------|
| COVID Crash (Mar 2020) | Oct-Nov 2008 GFC | Mostly negative | 0.0008–0.0035 (high) |
| AI Rally (Jun 2023) | 2010 recovery, 2016 steady | All positive | 0.00003–0.00012 (low) |
| Rate Hike Start (Jan 2022) | Dec 2010, Mar 2016, Mar-Apr 2018 | Mixed | Moderate |

**Key finding:** The encoder (trained on 2007-2019 only) generalizes out-of-sample — COVID crash correctly identifies 2008 GFC as its nearest neighbor. Euclidean distance in 1280D gives proper contrast (self-distance 0.019 vs next 4.26).

### JEPA Configuration

All parameters are configurable via environment variables (`JEPA_*` and `TRAIN_*` prefixes):

| Parameter | Default | Description |
| --- | --- | --- |
| `mask_ratio` | 0.7 | Fraction of patches masked during training |
| `num_patches` | 20 | Total patches per sample |
| `encoder_embed_dim` | 64 | Encoder embedding dimension |
| `predictor_embed_dim` | 128 | Predictor embedding dimension |
| `num_layers_encoder` | 4 | Transformer layers in encoder |
| `num_layers_predictor` | 2 | Transformer layers in predictor |
| `nhead_encoder` | 8 | Attention heads in encoder |
| `batch_size` | 4 | Training batch size |
| `lr` | 3e-4 | Learning rate |
| `lambda_v` | 2 | VICReg variance loss weight |
| `lambda_cv` | 2.1 | VICReg covariance loss weight |

---

## Project Structure

```
├── var_mpc/                    # Phase 1: VAR-MPC
│   ├── dynamics.py             # MPC Planner (OSQP solver) & Market Simulator
│   ├── VAR_setup.py            # VAR model training, forecasting, ADF checks
│   ├── file.py                 # Main backtest loop (2020-2025)
│   ├── investors.py            # Benchmark comparison
│   ├── min_variance.py         # Min-variance benchmark
│   ├── unit_tests.py           # Pytest suite
│   └── testcode.py             # Test code
│
├── jepa/                       # Phase 2: JEPA + KV Retrieval
│   ├── models/
│   │   ├── encoder.py          # Transformer encoder (49 → 64 dim)
│   │   ├── predictor.py        # Transformer predictor
│   │   ├── decoder.py          # Linear decoder
│   │   ├── tokenizer.py        # Conv1D tokenizer
│   │   └── utils/              # Mask utils, attention modules
│   ├── data/
│   │   ├── data_class_parquet.py  # StockMarketJEPADataset
│   │   ├── dataextraction.py      # DataExtractor
│   │   └── parquet_data/          # Training + test parquet files
│   ├── training/
│   │   ├── jepa-training.py       # JEPA training with VICReg
│   │   ├── test_jepa_embeddings.py # Evaluation suite
│   │   └── regime_probe.py        # Regime classifier probe
│   ├── embedding_db/
│   │   ├── cost_fcn.py            # Build embedding database
│   │   └── query.py               # KV retrieval + sparse attention
│   ├── mpc/                       # (empty) — future MPC integration
│   └── scripts/                   # (empty) — future scripts
│
├── jepa-model/                 # Model checkpoints
├── setup.py                    # Editable install (pip install -e .)
├── AGENTS.md                   # LLM codebase index instructions
├── lab-notes/                  # Experiment logs (auto-indexed on commit)
├── .codebase/                  # SQLite codebase index (auto-generated)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/adilfaisal01/portfolio-optimization.git
cd portfolio-optimization
pip install -e .        # editable install for clean imports
```

### Dependencies

- `numpy`, `pandas`, `scipy`
- `statsmodels` (VAR modeling)
- `osqp` (quadratic programming)
- `torch` (JEPA training)
- `yfinance` (benchmark data)
- `matplotlib` (visualization)
- `pytest` (testing)

---

## Usage

### Run the VAR-MPC Backtest

```bash
python var_mpc/file.py
```

This loads historical data (2009–2019 training, 2020–2025 test), trains the VAR model, runs the adaptive MPC loop, and generates equity curves and performance metrics.

### Build the Embedding Database

```bash
python jepa/embedding_db/cost_fcn.py
```

Runs the JEPA encoder over all 163 training windows, saves embeddings + returns + covariance to `jepa-model/analysis/iteration-1/`.

### Query Similar Regimes

```python
from jepa.embedding_db.query import weighted_financial_metrics

cov, ret = weighted_financial_metrics(query_embedding, k=5, tau=1.0)
# cov: (11, 11) — regime-conditional covariance
# ret: (11,)   — regime-conditional expected returns
```

### Train the JEPA Model

```bash
python jepa/training/jepa-training.py
```

Or configure via environment variables:

```bash
JEPA_MASK_RATIO=0.3 JEPA_NUM_PATCHES=30 TRAIN_NUM_EPOCHS=10 python jepa/training/jepa-training.py
```

### Run Tests

```bash
pytest var_mpc/unit_tests.py -v
```

---

## Vault Lab Notebooks

Detailed experiment logs, architecture decisions, and findings live in the Obsidian vault at `Projects/portfolio-optimization/`. Key entries:

| Date | Topic |
| --- | --- |
| 2026-05-26 | MPC-VAR + GRPO Foundations |
| 2026-06-08 | JEPA Integration — first training pipeline |
| 2026-06-10 | VICReg Regularization — preventing collapse |
| 2026-06-11 | JEPA Evaluation Autopsy — The Predictor Never Won |
| 2026-06-21 | MPC Friction Analysis & VectorBT Validation |
| 2026-06-24 | Regime Probe — JEPA Embeddings Encode Market Regime |
| 2026-07-06 | Embedding Database + KV Retrieval — COVID crash → 2008 GFC |

---

## Author

**Adil Faisal** — 2026

# Portfolio Optimization — VAR-MPC + JEPA

**Two-phase project: Model Predictive Control meets Self-Supervised Learning for portfolio management.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![OSQP](https://img.shields.io/badge/Solver-OSQP-green.svg) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)

---

## Overview

This project explores two complementary approaches to dynamic portfolio management:

1. **Phase 1 — VAR-MPC:** A model-based control approach using Vector Autoregression for return forecasting and Model Predictive Control for optimal rebalancing.
2. **Phase 2 — JEPA:** A self-supervised learning approach using a Joint-Embedding Predictive Architecture to learn market representations without labels, then probing what those representations encode.

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

## Phase 2: JEPA Representation Learning

### Motivation

The VAR-MPC approach relies on point forecasts from a linear model. What if we could learn richer representations of market state — capturing regime changes, volatility regimes, and latent market factors — without requiring labeled data?

### Architecture

The JEPA (Joint-Embedding Predictive Architecture) uses a Transformer-based encoder to embed market patches and a predictor to forecast masked patches in latent space:

```
Input patches (20 windows, 49 dim each)
  → Random masking (80% visible, 20% masked)
    → Encoder (Transformer, 4 layers, 64 embed dim) — processes visible patches only
    → Predictor (Transformer, 2 layers, 128 embed dim) — predicts masked patch embeddings
      → VICReg loss (variance + invariance + covariance) + L1 prediction loss
```

**Key components:**
- **Encoder:** 4-layer Transformer with Conv1D tokenizer, 64 embed dim, 8 attention heads
- **Predictor:** 2-layer Transformer with learned mask tokens, 128 embed dim, 4 attention heads
- **Decoder:** Simple linear layer for downstream forecasting tasks
- **Tokenizer:** Conv1D + linear projection to patch embeddings
- **Data:** `StockMarketJEPADataset` — parquet-based dataset with VIX conditioning

**Loss function:** VICReg (Variance-Invariance-Covariance Regularization) with `λ_v = 2` (variance), `λ_cv = 2.1` (covariance), plus L1 prediction loss on masked patches.

### What We Learned

#### 1. JEPA Embeddings Encode Market Regime

A linear probe (logistic regression) on averaged encoder embeddings achieved **97.3% full-train accuracy** classifying which year/regime a window belongs to — the confusion matrix was nearly diagonal with only 2 mistakes out of 74 windows.

Per-patch cross-validation accuracy reached **38.9%** — 2.3× random chance (16.7%), confirming regime signal is distributed across all 64 embedding dimensions rather than concentrated in a single "regime neuron."

**Full experiment log:** `lab-notes/daily/2026-06-24.md`.

#### 2. The Predictor Never Won

After testing 3 encoder architectures, 2 decoding strategies, and 6 market regimes: the JEPA predictor transformer never outperformed a simple 2-layer MLP decoder on multi-step feature forecasting. The encoder does the heavy lifting — how you decode matters less than what the encoder learns.

**Full autopsy:** `lab-notes/daily/2026-06-21.md` (see the JEPA Evaluation Autopsy section).

#### 3. VICReg Prevents Collapse

Without VICReg regularization, the JEPA embeddings collapsed to a low-rank subspace (~8 effective dimensions out of 64). With VICReg (variance + covariance terms), the effective rank increased to ~125 out of 896 dimensions — the representation spread out and captured more information.

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
├── dynamics.py              # MPC Planner (OSQP solver) & Market Simulator
├── VAR_setup.py             # VAR model training, forecasting, ADF checks
├── file.py                  # Main backtest loop (2020-2025)
├── paper_trading_bot.py     # Live paper trading bot (Alpaca)
├── cascading_mpc_v3.py      # SE701 daily MPC + intraday drift correction
├── backtest_covar.py        # Static vs per-step covariance comparison
├── unit_tests.py            # Pytest suite
├── investors.py             # Benchmark comparison
├── min_variance.py          # Min-variance benchmark
│
├── jepa-training.py         # JEPA training script (configurable via env vars)
├── regime_probe.py          # Regime classifier probe
├── src/
│   ├── models/
│   │   ├── encoder.py       # Transformer encoder
│   │   ├── predictor.py     # Transformer predictor
│   │   ├── decoder.py       # Linear decoder
│   │   └── tokenizer.py     # Conv1D tokenizer
│   └── data_loaders/
│       ├── data_loader.py   # JEPA data loaders
│       └── data_class.py    # CSV data loader
├── individual_stocks/
│   └── data_class_parquet.py # StockMarketJEPADataset (parquet)
│
├── AGENTS.md                # LLM codebase index instructions
├── lab-notes/
│   └── daily/               # Experiment logs (auto-indexed on commit)
├── .codebase/               # SQLite codebase index (auto-generated)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/adilfaisal01/portfolio-optimization.git
cd portfolio-optimization
pip install -r requirements.txt
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
python file.py
```

This loads historical data (2009–2019 training, 2020–2025 test), trains the VAR model, runs the adaptive MPC loop, and generates equity curves and performance metrics.

### Run the Paper Trading Bot

```bash
python paper_trading_bot.py
```

Runs daily at market open: pulls data from Alpaca/IEX, forecasts with VAR, solves MPC, and rebalances the portfolio.

### Train the JEPA Model

```bash
python jepa-training.py
```

Or configure via environment variables:

```bash
JEPA_MASK_RATIO=0.3 JEPA_NUM_PATCHES=30 TRAIN_NUM_EPOCHS=10 python jepa-training.py
```

### Run Tests

```bash
pytest unit_tests.py -v
```

---

## Vault Lab Notebooks

Detailed experiment logs, architecture decisions, and findings live in `lab-notes/daily/`. Key entries:

| Date | Topic |
| --- | --- |
| 2026-05-26 | MPC-VAR + GRPO Foundations |
| 2026-06-08 | JEPA Integration — first training pipeline |
| 2026-06-10 | VICReg Regularization — preventing collapse |
| 2026-06-11 | JEPA Evaluation Autopsy — The Predictor Never Won |
| 2026-06-21 | MPC Friction Analysis & VectorBT Validation |
| 2026-06-24 | Regime Probe — JEPA Embeddings Encode Market Regime |

---

## Future Work

- **Stochastic MPC** using a distribution of returns rather than point forecasts
- **Adaptive system identification** (Recursive Least Squares, windowing)
- **JEPA → MPC pipeline:** Use JEPA embeddings as input features for the MPC controller
- **Expand beyond 7 assets** to include individual stocks and alternatives
- **Online JEPA training** with streaming market data

---

## Author

**Adil Faisal** — 2026

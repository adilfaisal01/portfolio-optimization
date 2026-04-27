# MPC-VAR Portfolio Management

**Dynamic Portfolio Management using Model Predictive Control with Vector Autoregression**
  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  

![OSQP](https://img.shields.io/badge/Solver-OSQP-green.svg)

* * *

## Overview

This project implements a **Model Predictive Control (MPC)** framework for dynamic portfolio management across 7 ETFs, using a **Vector Autoregression (VAR)** model for return forecasting. The system rebalances a multi-asset portfolio daily over a 30-day planning horizon, with the VAR model re-fitted every 21 days to adapt to changing market dynamics.
The strategy was backtested from **January 2020 to December 2025** — a period spanning the COVID-19 crash, rising interest rates, and the AI-driven market rally — and benchmarked against buy-and-hold and prominent investors.

* * *

## Assets Managed

| Ticker | Description | Asset Class |
| --- | --- | --- |
| **AGG** | iShares Core U.S. Aggregate Bond ETF | Bonds |
| **SPY** | SPDR S&P 500 ETF | US Stocks (Large Cap) |
| **GLD** | SPDR Gold Shares | Precious Metals |
| **SLV** | iShares Silver Trust | Precious Metals |
| **VTI** | Vanguard Total Stock Market ETF | US Stocks (Total Market) |
| **VEA** | Vanguard FTSE Developed Markets ETF | International Stocks (Developed) |
| **VWO** | Vanguard FTSE Emerging Markets ETF | International Stocks (Emerging) |

* * *

## Project Structure

```
├── dynamics.py          # MPC Planner (OSQP solver) & Market Simulator
├── VAR_setup.py         # VAR model training, forecasting, and ADF stationarity checks
├── file.py              # Main backtest loop (adaptive refit, execution, visualization)
├── testcode.py          # Standalone prototype of the MPC QP formulation
├── min_variance.py      # Min-variance benchmark strategy (daily rolling)
├── investors.py         # Benchmark comparison vs ARKK, BRK-B, SPY, PSHZF
├── unit_tests.py        # Pytest suite covering planner, simulator, and edge cases
└── SE701_stage2_report_adil.pdf  # Full project report
```

* * *

## Methodology

### State-Space Model

The portfolio state is defined as:

```
x_k = [S_k, w_k]ᵀ
```

where `S_k` is the log-wealth and `w_k` is the vector of asset weights. The dynamics are:

```
x_{k+1} = A_k x_k + B u_k
```

With the linearized log-wealth form derived via Taylor series approximation:
| Matrix | Structure |
| --- | --- |
| **A** | `[[1, r_logᵀ], [0, I]]` |
| **B** | `[[0], [I]]` |

The log-returns `r_log` are forecast by the VAR model, making this a **Linear Time-Varying (LTV)** system.

### Cost Function

A multi-objective quadratic cost is optimized:

```
J = Σ ( w_kᵀ Σ w_k   +   u_kᵀ R u_k )   -   γ S_N
    └── risk ──┘     └─ transaction cost ─┘   └─ terminal wealth ─┘
```

*   **Risk** is penalized via the covariance matrix `Σ` from the VAR forecast
    
*   **Transaction costs** use a quadratic penalty scaled by the **CBOE Volatility Index (VIX)**:
    
    ```
    ρ = 2 × (VIX_t / VIX_normal)
    ```
    
    This increases cost during volatile periods (wider bid-ask spreads)
    
*   **Terminal wealth** maximization is controlled by risk aversion parameter `γ`
    

### Constraints

*   **Box constraint:** `0 ≤ w_i ≤ w_max` (no short selling, capped concentration)
    
*   **Sum-to-one:** `Σ w_i = 1` (fully invested)
    

### MPC Algorithm

1.  Obtain current state `x_0 = [S_t, w_t]ᵀ`
    
2.  Forecast returns and covariance using the VAR model
    
3.  Build the condensed dynamics matrices `T_bar`, `S_bar`
    
4.  Formulate and solve the **Quadratic Program (QP)** using **OSQP**
    
5.  Apply only the first control `u*_0` (receding horizon)
    
6.  Update wealth with realized returns; let weights drift with market movements
    

* * *

## Key Results

### Performance (2020–2025)

| Metric | Benchmark (Buy & Hold) | MPC (40% max weight) |
| --- | --- | --- |
| **CAGR** | 12.85% | **14.77%** |
| **Sharpe Ratio** | 0.61 | **0.73** |
| **Max Drawdown** | 25.34% | **26.37%** |
| **Calmar Ratio** | 0.51 | **0.56** |
| **Annual Volatility** | 14.83% | **16.53%** |
| **Mean Daily Turnover** | – | **0.47%** |

The **40% weight cap** achieved the best risk-return balance across all configurations tested.

### Comparison with Notable Investors

| Strategy | Sharpe | CAGR | Max Drawdown | Calmar |
| --- | --- | --- | --- | --- |
| **MPC (40%)** | **0.73** | 14.77% | 26.37% | **0.56** |
| ARKK (Cathie Wood) | 0.34 | 8.08% | 80.91% | 0.10 |
| BRK-B (Buffett) | 0.59 | 14.11% | 29.57% | 0.47 |
| SPY (S&P 500) | 0.63 | 14.97% | 33.72% | 0.44 |
| PSHZF (Pershing Square) | **0.79** | **23.38%** | 32.96% | **0.70** |

The MPC strategy delivers superior **risk-adjusted returns** — second-highest Sharpe ratio and best Calmar ratio among the compared strategies, while keeping drawdown lower than all benchmarks except its own variants.

* * *

## Installation

```bash
git clone https://github.com/yourusername/mpc-var-portfolio.git
cd mpc-var-portfolio
pip install -r requirements.txt
```

### Dependencies

*   `numpy`, `pandas`, `scipy`
    
*   `statsmodels` (for VAR modeling)
    
*   `osqp` (quadratic programming solver)
    
*   `yfinance` (for benchmark data)
    
*   `matplotlib` (visualization)
    
*   `pytest` (testing)
    

* * *

## Usage

### Run the Full Backtest

```bash
python file.py
```

This will:

1.  Load historical return data and VIX data
    
2.  Train the VAR model on data from **2009–2019**
    
3.  Run the adaptive MPC loop from **2020–2025**
    
4.  Generate equity curves, allocation plots, and performance metrics
    

### Run Tests

```bash
pytest unit_tests.py -v
```

Tests cover:

*   Planner solver compatibility with VAR output shapes
    
*   Drift-and-rebalance cycle correctness
    
*   Log-wealth to dollar consistency
    
*   Flash crash resilience (20% market drop)
    
*   No short-selling constraint enforcement
    
*   Risk aversion behavior (high variance avoidance)
    
*   Trade execution timing (trade-before-return ordering)
    

* * *

## Key Findings

1.  **40% weight cap** delivered the best trade-off: CAGR of **14.77%**, Sharpe of **0.73**, and Calmar of **0.56**
    
2.  **VIX-scaled transaction costs** kept mean daily turnover low at **0.47%**, preventing cost drag
    
3.  The MPC strategy outperformed buy-and-hold, ARKK, and Berkshire Hathaway on a risk-adjusted basis
    
4.  The strategy avoids speculative concentration — unlike ARKK's 80.91% drawdown during the 2022–2023 tech downturn
    

* * *

## Future Work

*   **Stochastic MPC** using a distribution of returns rather than point forecasts
    
*   **Adaptive system identification** (e.g., Recursive Least Squares, windowing) for more robust dynamics
    
*   Expand beyond 7 assets to include individual stocks and alternative asset classes
    

* * *

## Author

**Adil Faisal** — 2026
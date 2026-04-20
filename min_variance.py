import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from VAR_setup import VARAnalysis

# ─────────────────────────────────────────────
# 1. DATA SETUP
# ─────────────────────────────────────────────
transition_date = "2020-01-01"
var_eng = VARAnalysis(final_date=transition_date)
train_df, test_df = var_eng.data_segmentation()
indices = var_eng.indices_list

# Combine train + test into one dataframe for easy rolling slicing
all_data   = pd.concat([train_df, test_df], ignore_index=True)
train_end  = len(train_df)  # index where test period starts

# ─────────────────────────────────────────────
# 2. MIN-VAR SOLVER (pure — no cost term)
# ─────────────────────────────────────────────
def min_variance_weights(window_returns, w_prev):
    """
    Pure minimum variance solver.
    Costs applied post-hoc in the loop, not inside the objective.
    """
    cov = np.cov(window_returns.T)
    N   = window_returns.shape[1]

    def objective(w):
        return w.T @ cov @ w

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * N
    w0          = w_prev.copy()

    result = minimize(objective, w0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})

    if not result.success:
        print(f"Warning: Optimization failed — {result.message}. Holding previous weights.")
        return w_prev

    return result.x

# ─────────────────────────────────────────────
# 3. BACKTEST LOOP
# ─────────────────────────────────────────────
INITIAL_WEALTH  = 10000
TRANS_COST      = 1      # same R as MPC, applied post-hoc
ROLLING_WINDOW  = 252    # 1-year rolling window

minvar_wealth   = INITIAL_WEALTH
history         = []
weights_history = []

# Initial weights: solve on the last 252 days of training data
initial_window = all_data[indices].iloc[train_end - ROLLING_WINDOW : train_end].values
w_prev         = min_variance_weights(initial_window,
                                       w_prev=np.ones(len(indices)) / len(indices))

print("Starting Min-Variance Backtest: 2020 to 2025...")

for t in range(len(test_df)):

    # --- A. Rolling window: 252 days ending at today (uses only past data) ---
    window_data = all_data[indices].iloc[train_end + t - ROLLING_WINDOW : train_end + t].values
    w_current   = min_variance_weights(window_data, w_prev=w_prev)

    # --- B. Post-hoc transaction cost: same u^T R u as MPC ---
    u             = w_current - w_prev
    realised_cost = TRANS_COST * (u @ u)

    # --- C. Execute ---
    actual_r     = test_df[indices].iloc[t].values
    daily_return = actual_r @ w_current - realised_cost
    minvar_wealth *= np.exp(daily_return)

    # --- D. Record ---
    history.append({
        'Date':   test_df.iloc[t]['Date'],
        'Wealth': minvar_wealth
    })
    weights_history.append(w_current.copy())

    # --- E. Update ---
    w_prev = w_current.copy()

    print(t)

print("Backtest Complete!")

# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────
perf_df           = pd.DataFrame(history)
perf_df['Date']   = pd.to_datetime(perf_df['Date'])
perf_df['Return'] = perf_df['Wealth'].pct_change()

rf_daily   = 0.00012

sharpe     = (perf_df['Return'].mean() - rf_daily) / perf_df['Return'].std() * np.sqrt(252)
volatility = perf_df['Return'].std() * np.sqrt(252)

peak         = perf_df['Wealth'].cummax()
drawdown     = (perf_df['Wealth'] - peak) / peak
max_drawdown = abs(drawdown.min())

years        = (pd.to_datetime(test_df.iloc[-1]['Date']) - pd.to_datetime(test_df.iloc[0]['Date'])).days / 365.25
total_return = (minvar_wealth / INITIAL_WEALTH) - 1
cagr         = (1 + total_return) ** (1 / years) - 1
calmar       = cagr / max_drawdown

weight_data  = np.array(weights_history)
turnover     = np.abs(np.diff(weight_data, axis=0)).sum(axis=1)

print("\n── Min-Variance Results (daily rolling, post-hoc costs) ────")
print(f"Total Return:    {total_return:.2%}")
print(f"CAGR:            {cagr:.2%}")
print(f"Sharpe Ratio:    {sharpe:.2f}")
print(f"Max Drawdown:    {max_drawdown:.2%}")
print(f"Calmar Ratio:    {calmar:.2f}")
print(f"Annual Vol:      {volatility:.2%}")
print(f"Final Wealth:    ${minvar_wealth:,.0f}")
print(f"Mean Turnover:   {turnover.mean():.4f}")
print(f"Max Turnover:    {turnover.max():.4f}")

print("\nInitial Min-Var Weights:")
for asset, weight in zip(indices, weights_history[0]):
    print(f"  {asset}: {weight:.4f}")

print("\nAverage Weights Over Test Period:")
for asset, weight in zip(indices, weight_data.mean(axis=0)):
    print(f"  {asset}: {weight:.4f}")

# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────

plt.figure(figsize=(12, 6))
plt.plot(perf_df['Date'], perf_df['Wealth'], color='blue', label='Min-Variance (daily rolling)')
plt.title("Min-Variance Equity Curve: 2020–2025")
plt.xlabel("Year")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.stackplot(perf_df['Date'], weight_data.T, labels=indices)
plt.title("Min-Variance Portfolio Allocation: 2020–2025")
plt.ylabel("Weight")
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.fill_between(perf_df['Date'], drawdown, 0, color='red', alpha=0.4)
plt.title("Min-Variance Drawdown: 2020–2025")
plt.xlabel("Year")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VAR_setup import VARAnalysis
from dynamics import MPCPLanner,MarketSimulator

# 1. INITIALIZATION
# ---------------------------------------------------------
transition_date = "2020-01-01"
var_eng = VARAnalysis(final_date=transition_date)
train_df, test_df = var_eng.data_segmentation()

# Pre-train the "Brain" on the 11-year stable regime
var_eng.fit_model(train_df)

# Initialize the "Physical Reality"
# Starting with $10,000 and equal weights
market = MarketSimulator(initial_wealth=10000, initial_weights=[1/7]*7)

# Initialize the "Strategist"
planner = MPCPLanner(n_assets=7, wmax=0.30, N_horizon=30, trans_cost=1)

# Storage for performance analysis
history = []
indices = var_eng.indices_list

# 2. THE ADAPTIVE LOOP (2020 - 2025)
# ---------------------------------------------------------
print("Starting Backtest: 2020 to 2025...")

# We use an expanding window. Every step, we add the new day to our history.
full_history = train_df.copy()
step=0

for t in range(len(test_df)):
    # --- A. Context Gathering ---
    # Get the last k days of returns to feed the VAR forecast
    current_lookback = full_history[indices].values[-var_eng.k_ar:]
    
    # --- B. Adaptive Refit (Monthly / Every 21 Days) ---
    if t % 21== 0 and t > 0:
        var_eng.fit_model(full_history)
        # Optional: print(f"Model updated on {test_df.iloc[t]['Date']}")

    # --- C. Hallucination (Forecasting) ---
    # We take the first step of forecast_cov as discussed
    r_hat, r_cov = var_eng.forecast_model(lookbackdata=current_lookback, N_horizon=30)
    
    # --- D. Planning (Solving OSQP) ---
    x_0 = market.get_state()
    u_today = planner.solver(x_0, r_hat, r_cov)
    
    # --- E. Execution (The Reality) ---
    actual_r = test_df[indices].iloc[t].values
    market.step(u_today, actual_r)
    
    # --- F. Recording & Data Management ---
    history.append({
        'Date': test_df.iloc[t]['Date'],
        'Wealth': market.get_total_value(),
        'Weights': market.w.flatten()
    })
    
    # Append today's actual return to history so the VAR can "see" it tomorrow
    full_history = pd.concat([full_history, test_df.iloc[[t]]], ignore_index=True)
    print(t)

print("Backtest Complete!")

# 3. ANALYSIS & VISUALIZATION
# ---------------------------------------------------------
perf_df = pd.DataFrame(history)
perf_df['Date'] = pd.to_datetime(perf_df['Date'])

plt.figure(figsize=(12, 6))
plt.plot(perf_df['Date'], perf_df['Wealth'], label='MPC-VAR Strategy')
plt.title("Equity Curve: 2020 - 2025 (COVID & Post-COVID Adaptation)")
plt.xlabel("Year")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.show()
# Plotting the Weight Distribution over time
weight_data = np.array([h['Weights'] for h in history])
plt.figure(figsize=(12, 6))
plt.stackplot(perf_df['Date'], weight_data.T, labels=indices)
plt.title("Portfolio Allocation (2020 - 2025)")
plt.ylabel("Weight (%)")
plt.legend(loc='upper left')
plt.show()

# Simple Benchmark Loop
benchmark_wealth = 10000
for t in range(len(test_df)):
    actual_r = test_df[indices].iloc[t].values
    # Average return of all 7 assets
    daily_bench_return = np.mean(actual_r) 
    benchmark_wealth *= np.exp(daily_bench_return)
print(f"Benchmark Final Wealth: ${benchmark_wealth:,.2f}")

# 1. Calculate Daily Returns
perf_df['MPC_Return'] = perf_df['Wealth'].pct_change()
# Benchmark (1/7 hold) returns
bench_returns = test_df[indices].mean(axis=1) 

# 2. Risk-Free Rate (Assuming ~3% annual, or 0.00012 per day)
rf_daily = 0.00012

# 3. Calculate Sharpe Ratios
mpc_sharpe = (perf_df['MPC_Return'].mean() - rf_daily) / perf_df['MPC_Return'].std() * np.sqrt(252)
bench_sharpe = (bench_returns.mean() - rf_daily) / bench_returns.std() * np.sqrt(252)

print(f"MPC Sharpe Ratio: {mpc_sharpe:.2f}")
print(f"Benchmark Sharpe Ratio: {bench_sharpe:.2f}")

# 4. Volatility (Risk)
print(f"MPC Annual Volatility: {perf_df['MPC_Return'].std() * np.sqrt(252):.2%}")
print(f"Benchmark Annual Volatility: {bench_returns.std() * np.sqrt(252):.2%}")

# Calculate Max Drawdown
peak = perf_df['Wealth'].cummax()
drawdown = (perf_df['Wealth'] - peak) / peak
max_drawdown = abs(drawdown.min())

# Calculate CAGR
years = (pd.to_datetime(test_df.iloc[-1]['Date']) - pd.to_datetime(test_df.iloc[0]['Date'])).days / 365.25
total_return = (market.get_total_value() / 10000) - 1
cagr = (1 + total_return)**(1/years) - 1

calmar = cagr / max_drawdown

print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Calmar Ratio: {calmar:.2f}")
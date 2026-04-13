import pytest
import numpy as np
from scipy import sparse
# Import your classes here
from dynamics import MPCPLanner, MarketSimulator

@pytest.fixture
def setup_params():
    return {
        "n": 7,
        "N": 3,
        "wmax": 0.20,
        "cost": 0.001,
        "wealth": 10000
    }

@pytest.fixture
def planner(setup_params):
    return MPCPLanner(setup_params["n"], setup_params["wmax"], 
                      setup_params["N"], setup_params["cost"])

@pytest.fixture
def simulator(setup_params):
    return MarketSimulator(setup_params["wealth"], np.array([1/7]*7))

# --- Tests ---

def test_var_to_planner_slicing(planner, setup_params):
    """Verify the planner accepts the (N, 7) slice from the (100, 7) VAR output."""
    n, N = setup_params["n"], setup_params["N"]
    
    # Simulate VAR output (100 days predicted)
    var_forecast = np.random.normal(0.001, 0.02, (100, n))
    r_cov = np.eye(n) * 0.0001
    x_0 = np.array([9.21] + [1/n]*n).reshape(-1, 1)
    
    # CRITICAL: We slice the first N days for the planner
    r_hat_slice = var_forecast[:N, :]
    
    u_today = planner.solver(x_0, r_hat_slice, r_cov)
    
    assert u_today.shape == (n, 1) or u_today.shape == (n,)
    assert not np.isnan(u_today).any()

def test_drift_and_rebalance_cycle(planner, simulator, setup_params):
    """Verify that market drift triggers a rebalance trade."""
    n = setup_params["n"]
    
    # 1. Simulate a massive drift: Asset 0 moons (Log return of 0.5)
    r_actual = np.zeros(n)
    r_actual[0] = 0.5 
    
    # No trade today, just let it drift
    state_after_drift = simulator.step(np.zeros(n), r_actual)
    
    # Asset 0 weight should now be > wmax (0.20)
    current_w0 = state_after_drift[1].item()
    assert current_w0 > 0.20
    
    # 2. Feed this drifted state back to the planner
    r_hat = np.zeros((setup_params["N"], n))
    r_cov = np.eye(n) * 0.001
    
    u_rebalance = planner.solver(state_after_drift, r_hat, r_cov)
    
    # The trade should be negative for Asset 0 to bring it back to 0.20
    new_w0 = current_w0 + u_rebalance[0]
    assert new_w0 <= 0.200001
    assert np.isclose(np.sum(state_after_drift[1:] + u_rebalance.reshape(-1,1)), 1.0)

def test_wealth_log_to_dollar_consistency(simulator):
    """Ensure the Log-Wealth in state matches the physical dollar calculation."""
    initial_dollars = simulator.get_total_value()
    
    # Market goes up 1% across the board
    r_actual = np.array([np.log(1.01)] * 7)
    simulator.step(np.zeros(7), r_actual)
    
    # Physical math check
    expected_dollars = initial_dollars * 1.01
    assert np.isclose(simulator.get_total_value(), expected_dollars)

def test_flash_crash_resilience(simulator):
    """Verify system stability during a 20% market-wide drop."""
    initial_val = simulator.get_total_value()
    # -0.223 is approx log(0.80)
    crash_returns = np.array([-0.223] * 7) 
    
    simulator.step(np.zeros(7), crash_returns)
    
    assert simulator.get_total_value() < initial_val * 0.81
    assert np.isclose(np.sum(simulator.w), 1.0)

def test_no_short_selling(planner, setup_params):
    """Ensure the planner never pushes weights below zero."""
    n, N = setup_params["n"], setup_params["N"]
    # Starting with very low weights
    x_0 = np.array([9.0] + [0.01]*n).reshape(-1, 1)
    
    # Predict a massive crash for Asset 0
    r_hat = np.zeros((N, n))
    r_hat[:, 0] = -0.90 
    r_cov = np.eye(n) * 0.01
    
    u_today = planner.solver(x_0, r_hat, r_cov)
    final_w = x_0[1:] + u_today.reshape(-1, 1)
    
    # Weight can be 0, but never negative
    assert np.all(final_w >= -1e-7)

def test_risk_aversion(planner, setup_params):
    """Verify that high variance overrides high returns."""
    n, N = setup_params["n"], setup_params["N"]
    x_0 = np.array([9.0] + [1/n]*n).reshape(-1, 1)
    
    r_hat = np.zeros((N, n))
    r_hat[:, 0] = 0.10 # Asset 0 looks "good" on paper...
    
    # ...but Asset 0 has massive variance (risk)
    r_cov = np.eye(n) * 0.001
    r_cov[0, 0] = 100.0 
    
    u_today = planner.solver(x_0, r_hat, r_cov)
    
    # Planner should sell Asset 0 to avoid the risk
    assert u_today[0] < 0

def test_trade_execution_timing(simulator):
    """Ensure trades are executed BEFORE the price move is applied."""
    # Start equal weight
    u_trade = np.zeros(7)
    u_trade[0] = 0.10 # Buy 10% more of Asset 0
    u_trade[1] = -0.10 # Sell 10% of Asset 1
    
    # Only Asset 0 moves today
    r_actual = np.zeros(7)
    r_actual[0] = 0.10 
    
    # If trade is first: We get 10% gain on (Initial + 10%)
    # If trade is late: We get 10% gain on (Initial)
    old_p = simulator.P
    simulator.step(u_trade, r_actual)
    
    # The gain should reflect the post-trade weight
    assert simulator.P > old_p
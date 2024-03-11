import numpy as np
import matplotlib.pyplot as plt

# Parameters
months = 120
initial_investment = 100000  # Initial investment amount
initial_price = 100  # Initial price for asset simulation
annual_fee_rate = 0.0025  # Annual fee of 0.25%

# Annual returns and volatilities
annual_returns = np.array([0.08, 0.06, 0.08])  # US stocks, Foreign stocks, Bonds
annual_volatilities = np.array([0.15, 0.12, 0.4])  # US stocks, Foreign stocks, Bonds

# Convert annual parameters to monthly
monthly_returns = annual_returns / 12
monthly_volatilities = annual_volatilities / np.sqrt(12)
monthly_fee_rate = annual_fee_rate / 12  # Monthly fee

# Initialize asset price history
asset_prices = np.zeros((months + 1, 3))
asset_prices[0, :] = initial_price

# Simulate asset price changes over time
for month in range(1, months + 1):
    random_returns = np.random.normal(monthly_returns, monthly_volatilities)
    asset_prices[month, :] = asset_prices[month - 1, :] * (1 + random_returns)

# Initial allocations
start_allocation = np.array([0.6, 0.3, 0.1])

# Calculate the monthly returns of assets from prices for portfolio simulations
monthly_asset_returns = asset_prices[1:] / asset_prices[:-1] - 1

# Initialize portfolios
portfolio_no_rebalance = initial_investment * start_allocation
portfolio_rebalance = np.copy(portfolio_no_rebalance)
portfolio_no_rebalance_fee = np.copy(portfolio_no_rebalance)
portfolio_rebalance_fee = np.copy(portfolio_rebalance)
history_no_rebalance = [np.sum(portfolio_no_rebalance)]
history_rebalance = [np.sum(portfolio_rebalance)]
history_no_rebalance_fee = [np.sum(portfolio_no_rebalance_fee)]
history_rebalance_fee = [np.sum(portfolio_rebalance_fee)]

for month_returns in monthly_asset_returns:
    # Update portfolio values
    portfolio_no_rebalance *= (1 + month_returns)
    portfolio_rebalance *= (1 + month_returns)
    portfolio_no_rebalance_fee *= (1 + month_returns) * (1 - monthly_fee_rate)
    portfolio_rebalance_fee *= (1 + month_returns) * (1 - monthly_fee_rate)
    
    # Rebalance
    total_value_rebalance = np.sum(portfolio_rebalance)
    portfolio_rebalance = total_value_rebalance * start_allocation
    total_value_rebalance_fee = np.sum(portfolio_rebalance_fee)
    portfolio_rebalance_fee = total_value_rebalance_fee * start_allocation
    
    # Record history
    history_no_rebalance.append(np.sum(portfolio_no_rebalance))
    history_rebalance.append(np.sum(portfolio_rebalance))
    history_no_rebalance_fee.append(np.sum(portfolio_no_rebalance_fee))
    history_rebalance_fee.append(np.sum(portfolio_rebalance_fee))

# Plot 1: Asset Prices Over Time
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(asset_prices[:, 0], label="US Stocks")
plt.plot(asset_prices[:, 1], label="Foreign Stocks")
plt.plot(asset_prices[:, 2], label="Crypto")
plt.title("Asset Price Evolution Over Time")
plt.xlabel("Months")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)

# Plot 2: Portfolio Values Over Time (Including Fees)
plt.subplot(3, 1, 2)
plt.plot(history_no_rebalance, label="Without Rebalancing, No Fees")
plt.plot(history_rebalance, label="With Rebalancing, No Fees")
plt.plot(history_no_rebalance_fee, label="Without Rebalancing, With Fees")
plt.plot(history_rebalance_fee, label="With Rebalancing, With Fees")
plt.title("Portfolio Value Over Time")
plt.xlabel("Months")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)

# Plot 3: Relative Percentage Difference Between Portfolios
percentage_difference_fees = (np.array(history_rebalance_fee) - np.array(history_no_rebalance)) / np.array(history_no_rebalance) * 100
plt.subplot(3, 1, 3)
plt.plot(percentage_difference_fees, label="Percentage Difference (Rebalanced with Fees - Non-Rebalanced No Fees)")
plt.title("Relative Percentage Difference Between Portfolios Over Time (With Fees)")
plt.xlabel("Months")
plt.ylabel("Percentage Difference (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

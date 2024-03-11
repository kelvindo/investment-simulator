import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to simulate asset prices and calculate portfolio values
def simulate_assets_and_portfolios(months, initial_investment, initial_price, annual_returns, annual_volatilities, annual_fee_rate, start_allocation):
    monthly_returns = annual_returns / 12
    monthly_volatilities = annual_volatilities / np.sqrt(12)
    monthly_fee_rate = annual_fee_rate / 12

    asset_prices = np.zeros((months + 1, len(annual_returns)))
    asset_prices[0, :] = initial_price

    for month in range(1, months + 1):
        random_returns = np.random.normal(monthly_returns, monthly_volatilities)
        asset_prices[month, :] = asset_prices[month - 1, :] * (1 + random_returns)

    monthly_asset_returns = asset_prices[1:] / asset_prices[:-1] - 1

    portfolio_no_rebalance = initial_investment * start_allocation
    portfolio_rebalance = np.copy(portfolio_no_rebalance)
    portfolio_no_rebalance_fee = np.copy(portfolio_no_rebalance)
    portfolio_rebalance_fee = np.copy(portfolio_rebalance)
    
    history_no_rebalance = [np.sum(portfolio_no_rebalance)]
    history_rebalance = [np.sum(portfolio_rebalance)]
    history_no_rebalance_fee = [np.sum(portfolio_no_rebalance_fee)]
    history_rebalance_fee = [np.sum(portfolio_rebalance_fee)]

    for month_returns in monthly_asset_returns:
        portfolio_no_rebalance *= (1 + month_returns)
        portfolio_rebalance *= (1 + month_returns)
        portfolio_no_rebalance_fee *= (1 + month_returns) * (1 - monthly_fee_rate)
        portfolio_rebalance_fee *= (1 + month_returns) * (1 - monthly_fee_rate)

        total_value_rebalance = np.sum(portfolio_rebalance)
        portfolio_rebalance = total_value_rebalance * start_allocation
        total_value_rebalance_fee = np.sum(portfolio_rebalance_fee)
        portfolio_rebalance_fee = total_value_rebalance_fee * start_allocation

        history_no_rebalance.append(np.sum(portfolio_no_rebalance))
        history_rebalance.append(np.sum(portfolio_rebalance))
        history_no_rebalance_fee.append(np.sum(portfolio_no_rebalance_fee))
        history_rebalance_fee.append(np.sum(portfolio_rebalance_fee))
    
    return asset_prices, history_no_rebalance, history_rebalance, history_no_rebalance_fee, history_rebalance_fee

# Streamlit app
st.title('Portfolio Simulation: Vanguard vs. Wealthfront')

# Inputs
months = st.sidebar.slider('Months', 12, 240, 120)
initial_investment = st.sidebar.number_input('Initial Investment', value=100000)
annual_fee_rate = st.sidebar.number_input('Annual Fee Rate (%)', value=0.25) / 100
start_allocation = np.array([0.6, 0.3, 0.1])
annual_returns = np.array([0.08, 0.06, 0.08])
annual_volatilities = np.array([0.15, 0.12, 0.4])

if st.button('Run Simulation'):
    asset_prices, history_no_rebalance, history_rebalance, history_no_rebalance_fee, history_rebalance_fee = simulate_assets_and_portfolios(
        months, initial_investment, initial_price=100, annual_returns=annual_returns,
        annual_volatilities=annual_volatilities, annual_fee_rate=annual_fee_rate,
        start_allocation=start_allocation
    )

    # Always show the first plot if the simulation has been run
    fig1, ax1 = plt.subplots()
    for i, label in enumerate(["[60%] US Stocks", "[30%] Foreign Stocks", "[10%] Crypto"]):
        ax1.plot(asset_prices[:, i], label=label)
    ax1.set_title("Asset Price Evolution Over Time")
    ax1.set_xlabel("Months")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Expander for additional plots
    with st.expander("Show Results"):
        fig2, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 2: Portfolio Values Over Time (Including Fees)
        axs[0].plot(history_no_rebalance, label="Vanguard (No Rebalancing or Fees)")
        # axs[0].plot(history_rebalance, label="With Rebalancing, No Fees")
        # axs[0].plot(history_no_rebalance_fee, label="Without Rebalancing, With Fees")
        axs[0].plot(history_rebalance_fee, label="Wealthfront (With Rebalancing and Fees)")
        axs[0].set_title("Portfolio Value Over Time")
        axs[0].set_xlabel("Months")
        axs[0].set_ylabel("Portfolio Value ($)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 3: Relative Percentage Difference Between Portfolios
        percentage_difference_fees = (np.array(history_rebalance_fee) - np.array(history_no_rebalance)) / np.array(history_no_rebalance) * 100
        axs[1].plot(percentage_difference_fees, label="Percentage Difference (Wealthfront - Vanguard)")
        axs[1].set_title("Relative Percentage Differencee Between Portfolios (Wealthfront - Vanguard")
        axs[1].set_xlabel("Months")
        axs[1].set_ylabel("Percentage Difference (%)")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig2)

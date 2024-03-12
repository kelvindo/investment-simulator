import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go


# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data["Adj Close"]


# Function to normalize stock data
def normalize_stock_data(stock_data):
    return (stock_data / stock_data.iloc[0]) * 100


# Simulate portfolio performance with daily rebalancing
def simulate_portfolio(
    ticker1,
    ticker2,
    start_date,
    end_date,
    allocation1=0.5,
    allocation2=0.5,
    initial_investment=10000,
):
    # Fetch and normalize data
    stock1_data = normalize_stock_data(get_stock_data(ticker1, start_date, end_date))
    stock2_data = normalize_stock_data(get_stock_data(ticker2, start_date, end_date))

    # Combine into a single DataFrame
    df = pd.DataFrame({ticker1: stock1_data, ticker2: stock2_data})
    df.dropna(inplace=True)  # Ensure no NaN values

    # Calculate daily returns
    returns = df.pct_change().dropna()

    # Add returns to the df DataFrame
    df[f"{ticker1}_Return"] = returns[ticker1].rolling(window=30).mean()
    df[f"{ticker2}_Return"] = returns[ticker2].rolling(window=30).mean()

    # Initial investment split according to allocation
    investment_split = np.array([allocation1, allocation2]) * initial_investment

    # Arrays to hold portfolio values over time
    portfolio_values_no_rebalance = [initial_investment]
    portfolio_values_rebalance = [initial_investment]

    # Calculate portfolio value over time without rebalancing
    cumulative_returns_no_rebalance = (returns + 1).cumprod()
    portfolio_values_no_rebalance = cumulative_returns_no_rebalance.dot(
        investment_split
    )

    # Calculate portfolio value over time with daily rebalancing
    current_value = initial_investment
    for day_returns in returns.itertuples():
        # Calculate current portfolio value before rebalancing
        current_value *= 1 + np.dot(
            np.array(day_returns[1:]), np.array([allocation1, allocation2])
        )
        portfolio_values_rebalance.append(current_value)
        # Note: In a daily rebalancing scenario, the portfolio is rebalanced back to the original allocations every day,
        # so the allocations remain constant and are applied to the current portfolio value at the end of each day.

    # Convert portfolio values over time to Pandas Series for easy plotting
    portfolio_values_no_rebalance = pd.Series(
        portfolio_values_no_rebalance, index=returns.index
    )
    portfolio_values_rebalance = pd.Series(
        portfolio_values_rebalance[1:], index=returns.index
    )

    return df, portfolio_values_no_rebalance, portfolio_values_rebalance


# Streamlit UI (unchanged from your original script)
# Add your Streamlit user interface code here...


# Streamlit UI
st.title("Stock Rebalancing Simulator")

# User inputs
ticker1 = st.text_input("Enter the first stock ticker:", "META")
ticker2 = st.text_input("Enter the second stock ticker:", "NFLX")
start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-01-01"))

# Button to run simulation
if st.button("Simulate"):
    stock_data, portfolio_no_rebalance, portfolio_rebalance = simulate_portfolio(
        ticker1, ticker2, start_date, end_date
    )

    # Calculate the percent difference between the two portfolios
    percent_difference = (
        (portfolio_rebalance - portfolio_no_rebalance) / portfolio_no_rebalance
    ) * 100

    # Plot stock data
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stock_data.index, y=stock_data[ticker1], mode="lines", name=ticker1
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data.index, y=stock_data[ticker2], mode="lines", name=ticker2
        )
    )
    fig.update_layout(
        title="Normalized Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Normalized Close Price",
        legend_title="Ticker",
    )
    st.plotly_chart(fig)

    # Plot stock returns
    fig_returns = go.Figure()
    fig_returns.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data[f"{ticker1}_Return"],
            mode="lines",
            name=f"{ticker1} Returns",
        )
    )
    fig_returns.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data[f"{ticker2}_Return"],
            mode="lines",
            name=f"{ticker2} Returns",
        )
    )
    fig_returns.update_layout(
        title="Stock Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        legend_title="Ticker",
    )
    st.plotly_chart(fig_returns)

    # Plot portfolio values
    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_no_rebalance.index,
            y=portfolio_no_rebalance,
            mode="lines",
            name="No Rebalancing",
        )
    )
    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_rebalance.index,
            y=portfolio_rebalance,
            mode="lines",
            name="With Rebalancing",
        )
    )
    portfolio_fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend_title="Strategy",
    )
    st.plotly_chart(portfolio_fig)

    # Plot the percent difference
    percent_difference_fig = go.Figure()
    percent_difference_fig.add_trace(
        go.Scatter(
            x=percent_difference.index,
            y=percent_difference,
            mode="lines",
            name="Percent Difference (Rebalancing - No Rebalancing)",
        )
    )
    percent_difference_fig.update_layout(
        title="Percent Difference (Rebalancing - No Rebalancing) Over Time",
        xaxis_title="Date",
        yaxis_title="Percent Difference",
        legend_title="Difference",
    )
    st.plotly_chart(percent_difference_fig)

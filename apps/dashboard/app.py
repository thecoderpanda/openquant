"""OpenQuant India - interactive backtest dashboard.

Run with:

    uv run streamlit run apps/dashboard/app.py

Builds a strategy signal with ``oq-backtest`` primitives, runs an honest
gross-vs-net backtest, and renders the results as interactive charts:
equity curve, drawdown, and cost attribution. Falls back to a reproducible
synthetic price universe when no NSE data has been ingested locally.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oq_backtest import (
    PRESETS,
    backtest,
    equal_weight,
    mean_reversion_signal,
    momentum_signal,
    synthetic_universe,
)

try:
    from oq_data.api import list_symbols, wide_prices
except Exception:  # pragma: no cover - oq-data optional at runtime
    list_symbols = None
    wide_prices = None

st.set_page_config(page_title="OpenQuant India - Backtest Dashboard", layout="wide")

STRATEGIES = {
    "Momentum (top-K)": "momentum",
    "Mean reversion (bottom-K)": "mean_reversion",
    "Equal weight (buy & hold)": "equal_weight",
}


@st.cache_data(show_spinner=False)
def _load_real_symbols() -> list[str]:
    if list_symbols is None:
        return []
    try:
        return list_symbols()
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _synthetic_prices(n_symbols: int, n_days: int, seed: int) -> pd.DataFrame:
    return synthetic_universe(n_symbols=n_symbols, n_days=n_days, seed=seed)


@st.cache_data(show_spinner=False)
def _real_prices(symbols: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return wide_prices(symbols, start=start, end=end)


def _build_signal(
    strategy: str,
    prices: pd.DataFrame,
    lookback: int,
    top_k: int,
    schedule: str,
) -> pd.DataFrame:
    if strategy == "momentum":
        return momentum_signal(prices, lookback=lookback, top_k=top_k, schedule=schedule)
    if strategy == "mean_reversion":
        return mean_reversion_signal(prices, lookback=lookback, bottom_k=top_k, schedule=schedule)
    return equal_weight(prices.columns, prices.index)


st.title("OpenQuant India \u2014 Honest Backtest Dashboard")
st.caption("Your backtest is lying to you. This one shows the net number by default.")

with st.sidebar:
    st.header("Universe")
    real_symbols = _load_real_symbols()
    use_real_data = bool(real_symbols) and st.checkbox(
        f"Use ingested NSE data ({len(real_symbols)} symbols)", value=False
    )

    if use_real_data:
        chosen_symbols = st.multiselect(
            "Symbols", options=real_symbols, default=real_symbols[: min(20, len(real_symbols))]
        )
        date_range = st.date_input(
            "Date range",
            value=(pd.Timestamp.today() - pd.DateOffset(years=5), pd.Timestamp.today()),
        )
    else:
        st.caption("No local NSE data found \u2014 using a reproducible synthetic universe.")
        n_symbols = st.slider("Number of symbols", 5, 100, 30)
        n_days = st.slider("Number of trading days", 250, 3000, 1500, step=50)
        seed = st.number_input("Random seed", value=42, step=1)

    st.header("Strategy")
    strategy_label = st.selectbox("Signal", list(STRATEGIES.keys()))
    strategy = STRATEGIES[strategy_label]

    if strategy in ("momentum", "mean_reversion"):
        lookback = st.slider("Lookback (trading days)", 2, 252, 126 if strategy == "momentum" else 5)
        top_k = st.slider("Top / bottom K names", 1, 30, 5)
        schedule = st.selectbox("Rebalance schedule", ["daily", "weekly", "monthly", "quarterly"], index=2)
    else:
        lookback, top_k, schedule = 0, 0, "monthly"

    st.header("Costs & execution")
    cost_preset = st.selectbox("Broker cost preset", list(PRESETS.keys()), index=0)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 50.0, 5.0, step=0.5)
    initial_capital = st.number_input("Initial capital (INR)", value=1_000_000.0, step=100_000.0)

    run = st.button("Run backtest", type="primary", use_container_width=True)

if not run:
    st.info("Configure a strategy in the sidebar and click **Run backtest**.")
    st.stop()

with st.spinner("Running backtest..."):
    if use_real_data:
        if not chosen_symbols:
            st.error("Pick at least one symbol.")
            st.stop()
        start, end = date_range
        prices = _real_prices(tuple(chosen_symbols), str(start), str(end))
        if prices.empty:
            st.error("No price data for the selected symbols/date range.")
            st.stop()
    else:
        prices = _synthetic_prices(n_symbols, n_days, int(seed))

    signals = _build_signal(strategy, prices, lookback, top_k, schedule)
    result = backtest(
        signals,
        prices,
        costs=cost_preset,
        slippage=slippage_bps,
        initial_capital=initial_capital,
    )

summary = result.summary()

col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR (gross)", f"{summary['gross_cagr'] * 100:.2f}%")
col2.metric(
    "CAGR (net)",
    f"{summary['net_cagr'] * 100:.2f}%",
    delta=f"{(summary['net_cagr'] - summary['gross_cagr']) * 100:.2f} pp",
    delta_color="inverse",
)
col3.metric("Sharpe (net)", f"{summary['net_sharpe']:.2f}")
col4.metric("Max drawdown (net)", f"{summary['net_max_drawdown'] * 100:.2f}%")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Sortino (net)", f"{summary['net_sortino']:.2f}")
col6.metric("Calmar (net)", f"{summary['net_calmar']:.2f}")
col7.metric("Annual turnover", f"{summary['annual_turnover'] * 100:.1f}%")
col8.metric("Total cost paid (INR)", f"{summary['total_cost_inr']:,.0f}")

st.subheader("Equity curve: gross vs net")
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(x=result.gross_equity.index, y=result.gross_equity, name="Gross", line=dict(color="#9ca3af"))
)
eq_fig.add_trace(
    go.Scatter(x=result.net_equity.index, y=result.net_equity, name="Net (real)", line=dict(color="#16a34a"))
)
eq_fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Portfolio value (INR)")
st.plotly_chart(eq_fig, use_container_width=True)

col_dd, col_cost = st.columns(2)

with col_dd:
    st.subheader("Drawdown (net)")
    equity = result.net_equity.dropna()
    dd = equity / equity.cummax() - 1.0
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color="#dc2626")))
    dd_fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Drawdown (%)")
    st.plotly_chart(dd_fig, use_container_width=True)

with col_cost:
    st.subheader("Cost attribution")
    attribution = result.cost_attribution()
    cost_fig = go.Figure(go.Bar(x=attribution.index, y=attribution.values, marker_color="#2563eb"))
    cost_fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="INR over backtest")
    st.plotly_chart(cost_fig, use_container_width=True)

st.subheader("Text tearsheet")
st.code(result.tearsheet(), language=None)

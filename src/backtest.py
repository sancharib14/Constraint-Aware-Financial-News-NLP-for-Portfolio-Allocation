import pandas as pd


def run_backtest(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    weight_col: str = "weight",
    forward_return_col: str = "fwd_ret_1d"
) -> pd.DataFrame:
    """
    Run simple one-period-ahead backtest.

    weights on date t are multiplied by forward return from t to t+1.

    Returns:
        DataFrame with:
        date, portfolio_return
    """

    needed_cols = [date_col, ticker_col, forward_return_col]
    returns = panel[needed_cols].copy()

    merged = weights.merge(
        returns,
        on=[date_col, ticker_col],
        how="left"
    )

    merged["pnl_contribution"] = (
        merged[weight_col] * merged[forward_return_col]
    )

    portfolio_returns = (
        merged.groupby(date_col)["pnl_contribution"]
        .sum()
        .reset_index()
        .rename(columns={"pnl_contribution": "portfolio_return"})
    )

    portfolio_returns = portfolio_returns.sort_values(date_col).reset_index(drop=True)

    return portfolio_returns


def compute_turnover(
    weights: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    weight_col: str = "weight"
) -> pd.DataFrame:
    """
    Compute daily portfolio turnover.

    Turnover = sum absolute change in weights across tickers.

    We fill missing positions with zero.
    """

    wide = (
        weights.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=weight_col,
            aggfunc="sum"
        )
        .fillna(0.0)
        .sort_index()
    )

    turnover = wide.diff().abs().sum(axis=1)
    turnover.iloc[0] = wide.iloc[0].abs().sum()

    return turnover.reset_index().rename(columns={0: "turnover"})


def apply_transaction_costs(
    portfolio_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_bps: float = 5.0,
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Subtract transaction costs from portfolio returns.

    cost_bps = cost in basis points per unit turnover.
    5 bps means 0.0005 per 1.0 turnover.
    """

    df = portfolio_returns.merge(turnover, on=date_col, how="left")
    df["turnover"] = df["turnover"].fillna(0.0)

    cost_rate = cost_bps / 10000.0
    df["transaction_cost"] = cost_rate * df["turnover"]

    df["portfolio_return_net"] = (
        df["portfolio_return"] - df["transaction_cost"]
    )

    return df

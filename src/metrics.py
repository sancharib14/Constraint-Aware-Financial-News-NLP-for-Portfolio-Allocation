import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized return from periodic returns.

    Example:
    If returns are daily, periods_per_year = 252.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    cumulative_return = (1.0 + returns).prod()
    num_periods = len(returns)

    return cumulative_return ** (periods_per_year / num_periods) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized volatility from periodic returns.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.

    risk_free_rate should be annualized.
    For now we use 0.0 for simplicity.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    excess_returns = returns - risk_free_rate / periods_per_year
    vol = excess_returns.std()

    if vol == 0 or np.isnan(vol):
        return np.nan

    return excess_returns.mean() / vol * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from return series.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    return drawdown.min()


def summarize_returns(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """
    Return a dictionary of common performance metrics.
    """
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }

import numpy as np
import pandas as pd


def add_random_signal(
    panel: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Add a random signal for baseline testing.

    This should have no real predictive power.
    It is useful as a sanity check.
    """

    df = panel.copy()

    rng = np.random.default_rng(seed)
    df["signal_random"] = rng.normal(loc=0.0, scale=1.0, size=len(df))

    return df


def add_momentum_signal(
    panel: pd.DataFrame,
    lookback: int = 5
) -> pd.DataFrame:
    """
    Add simple past-return momentum signal.

    signal_momentum = past lookback-day return.

    Positive value means stock recently went up.
    Negative value means stock recently went down.
    """

    df = panel.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df[f"signal_mom_{lookback}d"] = (
        df.groupby("ticker")["close"].pct_change(periods=lookback)
    )

    return df


def add_reversal_signal(
    panel: pd.DataFrame,
    lookback: int = 5
) -> pd.DataFrame:
    """
    Add simple reversal signal.

    This is just negative momentum.

    If a stock went up recently, reversal signal says short it.
    If a stock went down recently, reversal signal says buy it.
    """

    df = panel.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    momentum = df.groupby("ticker")["close"].pct_change(periods=lookback)
    df[f"signal_rev_{lookback}d"] = -momentum

    return df

def add_llm_like_signal(
    panel: pd.DataFrame,
    base_signal_col: str = "signal_mom_5d",
    noise_scale: float = 0.02,
    confidence_scale: float = 5.0,
    seed: int = 123
) -> pd.DataFrame:
    """
    Create a synthetic LLM-like signal.

    This is not a real LLM signal.

    Purpose:
        Test the agentic portfolio framework before using real LLM/news data.

    Logic:
        signal_llm_like = confidence_scale * base_signal + noise

    Interpretation:
        A larger confidence_scale makes the signal more aggressive,
        similar to an overconfident LLM allocator.
    """

    df = panel.copy()

    if base_signal_col not in df.columns:
        raise ValueError(f"{base_signal_col} not found in dataframe.")

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=len(df)
    )

    df["signal_llm_like"] = (
        confidence_scale * df[base_signal_col].fillna(0.0) + noise
    )

    return df

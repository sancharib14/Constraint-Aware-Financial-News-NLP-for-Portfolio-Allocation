import pandas as pd


def make_long_short_weights(
    signal_df: pd.DataFrame,
    signal_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    top_k: int = 5,
    bottom_k: int = 5,
    gross_exposure: float = 1.0
) -> pd.DataFrame:
    """
    Create equal-weight long-short portfolio.

    For each date:
    - long top_k stocks by signal
    - short bottom_k stocks by signal

    gross_exposure = sum(abs(weights)).
    If gross_exposure = 1.0:
        long side = +0.5 total
        short side = -0.5 total
    """

    all_weights = []

    for date, group in signal_df.groupby(date_col):
        group = group.dropna(subset=[signal_col]).copy()

        if len(group) < top_k + bottom_k:
            continue

        group = group.sort_values(signal_col)

        short_names = group.head(bottom_k)
        long_names = group.tail(top_k)

        long_weight = gross_exposure / 2.0 / top_k
        short_weight = -gross_exposure / 2.0 / bottom_k

        for _, row in long_names.iterrows():
            all_weights.append({
                date_col: date,
                ticker_col: row[ticker_col],
                "weight": long_weight
            })

        for _, row in short_names.iterrows():
            all_weights.append({
                date_col: date,
                ticker_col: row[ticker_col],
                "weight": short_weight
            })

    weights = pd.DataFrame(all_weights)

    if len(weights) == 0:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight"])

    return weights.sort_values([date_col, ticker_col]).reset_index(drop=True)


def make_signal_weighted_long_short_weights(
    signal_df: pd.DataFrame,
    signal_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    top_k: int = 5,
    bottom_k: int = 5,
    gross_exposure: float = 1.0,
    min_abs_signal: float = 1e-8
) -> pd.DataFrame:
    """
    Create signal-weighted long-short portfolio.

    For each date:
    - long top_k stocks
    - short bottom_k stocks
    - allocate larger weights to stronger absolute signals

    This is closer to an LLM confidence-based allocator:
    high-confidence names receive larger positions.
    """

    all_weights = []

    for date, group in signal_df.groupby(date_col):
        group = group.dropna(subset=[signal_col]).copy()

        if len(group) < top_k + bottom_k:
            continue

        group = group.sort_values(signal_col)

        short_names = group.head(bottom_k).copy()
        long_names = group.tail(top_k).copy()

        # Long weights proportional to positive signal strength
        long_strength = long_names[signal_col].abs()
        long_strength = long_strength.clip(lower=min_abs_signal)

        # Short weights proportional to negative signal strength
        short_strength = short_names[signal_col].abs()
        short_strength = short_strength.clip(lower=min_abs_signal)

        long_weights = long_strength / long_strength.sum() * (gross_exposure / 2.0)
        short_weights = -short_strength / short_strength.sum() * (gross_exposure / 2.0)

        for idx, row in long_names.iterrows():
            all_weights.append({
                date_col: date,
                ticker_col: row[ticker_col],
                "weight": long_weights.loc[idx]
            })

        for idx, row in short_names.iterrows():
            all_weights.append({
                date_col: date,
                ticker_col: row[ticker_col],
                "weight": short_weights.loc[idx]
            })

    weights = pd.DataFrame(all_weights)

    if len(weights) == 0:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight"])

    return weights.sort_values([date_col, ticker_col]).reset_index(drop=True)

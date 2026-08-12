import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def directional_accuracy(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str
) -> float:
    """
    Fraction of times signal sign matches future return sign.

    Positive signal should predict positive return.
    Negative signal should predict negative return.
    """

    temp = df[[signal_col, return_col]].dropna().copy()

    if len(temp) == 0:
        return np.nan

    signal_direction = np.sign(temp[signal_col])
    return_direction = np.sign(temp[return_col])

    valid = signal_direction != 0

    if valid.sum() == 0:
        return np.nan

    return float((signal_direction[valid] == return_direction[valid]).mean())


def rank_ic_by_date(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Compute daily cross-sectional Spearman rank IC.

    For each date:
        correlation(rank(signal), rank(future_return))
    """

    rows = []

    for date, group in df.groupby(date_col):
        temp = group[[signal_col, return_col]].dropna()

        if len(temp) < 3:
            continue

        ic, pval = spearmanr(temp[signal_col], temp[return_col])

        rows.append({
            date_col: date,
            "rank_ic": ic,
            "p_value": pval,
            "n": len(temp)
        })

    return pd.DataFrame(rows)


def summarize_rank_ic(
    ic_df: pd.DataFrame
) -> dict:
    """
    Summarize rank IC time series.
    """

    if len(ic_df) == 0:
        return {
            "mean_rank_ic": np.nan,
            "median_rank_ic": np.nan,
            "ic_positive_rate": np.nan,
            "num_dates": 0,
        }

    return {
        "mean_rank_ic": float(ic_df["rank_ic"].mean()),
        "median_rank_ic": float(ic_df["rank_ic"].median()),
        "ic_positive_rate": float((ic_df["rank_ic"] > 0).mean()),
        "num_dates": int(len(ic_df)),
    }


def bucket_return_analysis(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    num_buckets: int = 5
) -> pd.DataFrame:
    """
    Sort observations into signal buckets and compute future returns.

    Bucket 0 = lowest signal
    Bucket num_buckets - 1 = highest signal
    """

    temp = df[[signal_col, return_col]].dropna().copy()

    temp["bucket"] = pd.qcut(
        temp[signal_col],
        q=num_buckets,
        labels=False,
        duplicates="drop"
    )

    bucket_stats = (
        temp.groupby("bucket")
        .agg(
            avg_signal=(signal_col, "mean"),
            avg_future_return=(return_col, "mean"),
            median_future_return=(return_col, "median"),
            count=(return_col, "count"),
        )
        .reset_index()
    )

    return bucket_stats


def top_minus_bottom_return(
    bucket_stats: pd.DataFrame
) -> float:
    """
    Difference between highest-signal bucket return and lowest-signal bucket return.
    """

    if len(bucket_stats) < 2:
        return np.nan

    bottom = bucket_stats.iloc[0]["avg_future_return"]
    top = bucket_stats.iloc[-1]["avg_future_return"]

    return float(top - bottom)

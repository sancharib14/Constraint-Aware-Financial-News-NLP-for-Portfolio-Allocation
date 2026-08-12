from typing import List

import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Download daily price data from yfinance.

    Returns:
        DataFrame with columns:
        date, ticker, close
    """

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker"
    )

    if raw.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    rows = []

    # Case 1: multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        available_tickers = raw.columns.get_level_values(0).unique()

        for ticker in tickers:
            if ticker not in available_tickers:
                print(f"Warning: {ticker} not found in downloaded data.")
                continue

            temp = raw[ticker].copy()
            temp = temp.reset_index()

            # yfinance can call the date column Date, Datetime, or something else
            date_col = temp.columns[0]

            if "Close" not in temp.columns:
                print(f"Warning: Close column missing for {ticker}.")
                continue

            temp = temp[[date_col, "Close"]].copy()
            temp.columns = ["date", "close"]
            temp["ticker"] = ticker

            rows.append(temp)

    # Case 2: single ticker
    else:
        ticker = tickers[0]

        temp = raw.copy()
        temp = temp.reset_index()

        date_col = temp.columns[0]

        if "Close" not in temp.columns:
            raise ValueError("Close column missing from downloaded data.")

        temp = temp[[date_col, "Close"]].copy()
        temp.columns = ["date", "close"]
        temp["ticker"] = ticker

        rows.append(temp)

    if len(rows) == 0:
        raise ValueError("No valid ticker data found after processing yfinance output.")

    prices = pd.concat(rows, ignore_index=True)

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.dropna(subset=["close"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices[["date", "ticker", "close"]]


def add_return_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add return and volatility features.

    Input:
        date, ticker, close

    Output:
        ret_1d
        fwd_ret_1d
        fwd_ret_5d
        vol_20d
        vol_20d_ann
    """

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily realized return from previous close to current close
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    # Next-day forward return, used as prediction target
    df["fwd_ret_1d"] = df.groupby("ticker")["ret_1d"].shift(-1)

    # Next 5-day forward return
    df["fwd_ret_5d"] = (
        df.groupby("ticker")["close"].shift(-5) / df["close"] - 1.0
    )

    # Trailing 20-day realized volatility
    df["vol_20d"] = (
        df.groupby("ticker")["ret_1d"]
        .rolling(window=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Annualized version
    df["vol_20d_ann"] = df["vol_20d"] * (252 ** 0.5)

    return df.reset_index(drop=True)


def build_price_panel(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Full price panel builder.

    Downloads prices and adds return features.
    """

    prices = download_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )

    panel = add_return_features(prices)

    return panel

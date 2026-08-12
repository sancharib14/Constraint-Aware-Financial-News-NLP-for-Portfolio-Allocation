import pandas as pd


def load_analyst_ratings_news(path: str) -> pd.DataFrame:
    """
    Load analyst_ratings_processed.csv and standardize it.

    Input columns:
        title, date, stock

    Output columns:
        date, ticker, text
    """

    df = pd.read_csv(path)

    df = df[["date", "stock", "title"]].copy()
    df.columns = ["date", "ticker", "text"]

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
        errors="coerce",
        format="mixed"
    ).dt.tz_convert(None).dt.normalize()

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["text"] = df["text"].astype(str)

    df = df.dropna(subset=["date", "ticker", "text"])
    df = df.drop_duplicates()

    return df.reset_index(drop=True)

def filter_news_to_universe(
    news: pd.DataFrame,
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Keep only selected tickers and date range.
    """

    df = news.copy()

    tickers = [t.upper() for t in tickers]

    df = df[df["ticker"].isin(tickers)].copy()
    df = df[df["date"] >= pd.to_datetime(start_date)]
    df = df[df["date"] < pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def aggregate_daily_news(
    news: pd.DataFrame,
    max_items_per_day: int = 10
) -> pd.DataFrame:
    """
    Combine multiple headlines for same ticker-date into one text block.

    Output:
        date, ticker, text, num_headlines
    """

    df = news.copy()
    df = df.sort_values(["date", "ticker"])

    daily = (
        df.groupby(["date", "ticker"])
        .agg(
            text=("text", lambda x: " ".join(x.head(max_items_per_day))),
            num_headlines=("text", "count")
        )
        .reset_index()
    )

    return daily


def merge_news_with_price_panel(
    news_daily: pd.DataFrame,
    price_panel: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge daily news with price panel.

    Keeps only ticker-date rows where news is available.
    """

    panel = price_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ticker"] = panel["ticker"].astype(str).str.upper()

    news = news_daily.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news["ticker"] = news["ticker"].astype(str).str.upper()

    merged = panel.merge(
        news,
        on=["date", "ticker"],
        how="inner"
    )

    return merged.reset_index(drop=True)

"""VIX and reproducible headline-only news sentiment descriptors."""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import os
import re

import numpy as np
import pandas as pd
import requests
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv

from src.security import redact_secrets

VIX_URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
POSITIVE = frozenset({"beats", "beat", "growth", "surge", "surges", "upgrade", "upgrades",
                      "raises", "raised", "record", "profit", "profits", "strong", "gain", "gains"})
NEGATIVE = frozenset({"misses", "miss", "decline", "declines", "downgrade", "downgrades",
                      "cuts", "cut", "loss", "losses", "weak", "lawsuit", "falls", "fall"})


def fetch_vix(timeout: int = 30) -> pd.DataFrame:
    response = requests.get(VIX_URL, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = frame.columns.str.strip().str.lower()
    date_column = "date" if "date" in frame else frame.columns[0]
    close_column = "close" if "close" in frame else "vix close"
    if close_column not in frame:
        raise ValueError(f"Cboe CSV has no VIX close: {list(frame.columns)}")
    out = pd.DataFrame({"observation_date": pd.to_datetime(frame[date_column], utc=True),
                        "vix": pd.to_numeric(frame[close_column], errors="coerce")}).dropna()
    out = out.sort_values("observation_date").drop_duplicates("observation_date")
    # Daily final close is conservatively available after that US session.
    session_date = out.observation_date.dt.tz_localize(None).dt.tz_localize("America/New_York")
    out["available_at"] = (session_date + pd.Timedelta(hours=20)).dt.tz_convert("UTC")
    out["vix_change_1d"] = out.vix.diff()
    out["vix_change_5d"] = out.vix.diff(5)
    mean = out.vix.rolling(252, min_periods=126).mean()
    std = out.vix.rolling(252, min_periods=126).std()
    out["vix_zscore_252d"] = (out.vix - mean) / std.replace(0, np.nan)
    return out


def headline_score(headline: str) -> float | None:
    """Fixed English keyword balance; no model or free-form LLM judgement."""
    words = set(re.findall(r"[a-z]+", headline.lower()))
    positive, negative = len(words & POSITIVE), len(words & NEGATIVE)
    return (positive - negative) / (positive + negative) if positive + negative else None


def fetch_alpaca_news(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Missing Alpaca credentials")
    request = NewsRequest(symbols=ticker, start=start, end=end, sort="asc",
                          include_content=False)
    try:
        news = NewsClient(api_key=key, secret_key=secret).get_news(request)
    except Exception as error:
        raise RuntimeError(f"Alpaca news request failed for {ticker}: "
                           f"{redact_secrets(str(error), (key, secret))}") from None
    articles = news.data.get("news", []) if hasattr(news, "data") else news.get("news", [])
    records = []
    for article in articles:
        item = article if isinstance(article, dict) else article.__dict__
        published = pd.Timestamp(item["created_at"]).tz_convert("UTC")
        updated = pd.Timestamp(item["updated_at"]).tz_convert("UTC")
        headline = item.get("headline") or ""
        records.append({"ticker": ticker, "published_at": published,
                        "available_at": max(published, updated),
                        "headline": headline, "summary": item.get("summary") or "",
                        "source": item.get("source") or "", "score": headline_score(headline),
                        "scoring_method": "fixed_english_headline_lexicon_v1"})
    return pd.DataFrame(records, columns=["ticker", "published_at", "available_at", "headline", "summary",
                                          "source", "score", "scoring_method"])


def news_snapshot(news: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> dict:
    as_of = pd.Timestamp(as_of)
    subset = news.loc[news.ticker.eq(ticker) & news.published_at.le(as_of) &
                      news.available_at.le(as_of)]
    one = subset.loc[subset.published_at.gt(as_of - pd.Timedelta(days=1))]
    seven = subset.loc[subset.published_at.gt(as_of - pd.Timedelta(days=7))]
    return {"news_count_1d": len(one), "news_count_7d": len(seven),
            "news_intensity_7d_per_day": len(seven) / 7,
            "news_sentiment_1d": float(one.score.mean()) if one.score.notna().any() else None,
            "news_sentiment_7d": float(seven.score.mean()) if seven.score.notna().any() else None,
            "news_scored_7d": int(seven.score.notna().sum()),
            "news_scoring_method": "fixed_english_headline_lexicon_v1"}

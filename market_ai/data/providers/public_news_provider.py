from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from urllib.parse import urlencode
from xml.etree import ElementTree

import pandas as pd
import requests


NEWS_TOPIC_QUERIES = {
    "energy": ('"crude oil" OR OPEC OR "oil prices" OR "natural gas"'),
    "metals": ('"gold prices" OR "silver prices" OR "copper prices"'),
    "fx_macro": ('"US dollar" OR "Federal Reserve" OR "Treasury yields" OR "Korean won"'),
    "equities_vol": ('"S&P 500" OR Nasdaq OR VIX OR "equity volatility"'),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return pd.Timestamp(parsed).tz_convert("UTC")


def _read_url(url: str) -> bytes:
    response = requests.get(
        url,
        headers={"User-Agent": "market-ai-data-collector/1.0"},
        timeout=15,
        verify=_requests_verify(),
    )
    response.raise_for_status()
    return response.content


def yahoo_finance_rss_url(symbol: str) -> str:
    return "https://feeds.finance.yahoo.com/rss/2.0/headline?" + urlencode(
        {"s": symbol, "region": "US", "lang": "en-US"}
    )


def fetch_yahoo_finance_rss(symbol: str) -> pd.DataFrame:
    payload = _read_url(yahoo_finance_rss_url(symbol))
    root = ElementTree.fromstring(payload)
    rows: list[dict[str, object]] = []
    for item in root.findall("./channel/item"):
        published = _parse_time(item.findtext("pubDate"))
        title = (item.findtext("title") or "").strip()
        if published is None or not title:
            continue
        rows.append(
            {
                "published_at": published.isoformat(),
                "symbol": symbol,
                "headline": title,
                "body": (item.findtext("description") or "").strip(),
                "source": "yahoo_finance_rss",
                "url": (item.findtext("link") or "").strip(),
                "retrieved_at": _utc_now_iso(),
            }
        )
    return pd.DataFrame(rows)


def gdelt_doc_url(query: str, *, timespan: str, maxrecords: int) -> str:
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(
        {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "timespan": timespan,
            "maxrecords": maxrecords,
        }
    )


def fetch_gdelt_articles(topic: str, query: str, *, timespan: str = "30d", maxrecords: int = 75) -> pd.DataFrame:
    payload = _read_url(gdelt_doc_url(query, timespan=timespan, maxrecords=maxrecords))
    data = json.loads(payload.decode("utf-8"))
    articles = data.get("articles", []) if isinstance(data, dict) else []
    rows: list[dict[str, object]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        published = _parse_time(str(article.get("seendate") or article.get("datetime") or ""))
        if published is None or not title:
            continue
        domain = str(article.get("domain") or article.get("source") or "gdelt").strip()
        rows.append(
            {
                "published_at": published.isoformat(),
                "symbol": "ALL",
                "headline": title,
                "body": "",
                "source": f"gdelt_doc_api:{topic}:{domain}",
                "url": str(article.get("url") or "").strip(),
                "retrieved_at": _utc_now_iso(),
            }
        )
    return pd.DataFrame(rows)


def normalize_public_news(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    columns = ["published_at", "symbol", "headline", "body", "source", "url", "retrieved_at"]
    if not valid:
        return pd.DataFrame(columns=columns)
    out = pd.concat(valid, ignore_index=True)
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    out["retrieved_at"] = pd.to_datetime(out["retrieved_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["published_at", "headline"])
    out["headline"] = out["headline"].astype(str).str.strip()
    out["url"] = out["url"].astype(str).str.strip()
    out = out[out["headline"] != ""]
    out = out.drop_duplicates(subset=["published_at", "headline", "url", "symbol"], keep="last")
    return out.sort_values("published_at")[columns].reset_index(drop=True)


def _requests_verify() -> str | bool:
    try:
        import certifi

        return certifi.where()
    except Exception:
        return True

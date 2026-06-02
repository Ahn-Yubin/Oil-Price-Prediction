from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
from urllib.parse import urlencode
from xml.etree import ElementTree

import pandas as pd
import requests


NEWS_TOPIC_QUERIES = {
    "energy": ('("crude oil" OR OPEC OR "oil prices" OR "natural gas") sourcelang:eng'),
    "metals": ('("gold prices" OR "silver prices" OR "copper prices") sourcelang:eng'),
    "fx_macro": ('("US dollar" OR "Federal Reserve" OR "Treasury yields" OR "Korean won") sourcelang:eng'),
    "equities_vol": ('("S&P 500" OR Nasdaq OR VIX OR "equity volatility") sourcelang:eng'),
}

GOOGLE_NEWS_TOPIC_QUERIES = {
    "energy": '"crude oil" OR OPEC OR "oil prices" OR "natural gas"',
    "metals": '"gold prices" OR "silver prices" OR "copper prices"',
    "fx_macro": '"US dollar" OR "Federal Reserve" OR "Treasury yields" OR "Korean won"',
    "equities_vol": '"S&P 500" OR Nasdaq OR VIX OR "equity volatility"',
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 14 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed)
    if len(text) == 16 and text[8] == "T" and text.endswith("Z"):
        parsed = pd.to_datetime(text, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed)
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return pd.Timestamp(parsed).tz_convert("UTC")


def _read_url(
    url: str,
    *,
    retries: int = 1,
    backoff_seconds: float = 2.0,
    sleep_seconds: float = 0.0,
    timeout_seconds: float = 8.0,
) -> bytes:
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    last_error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        response = requests.get(
            url,
            headers={"User-Agent": "market-ai-data-collector/1.0"},
            timeout=timeout_seconds,
            verify=_requests_verify(),
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.content
        last_error = requests.HTTPError(f"429 Too Many Requests for url: {url}", response=response)
        if attempt >= retries:
            break
        retry_after = response.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else backoff_seconds * (attempt + 1)
        except ValueError:
            wait = backoff_seconds * (attempt + 1)
        time.sleep(max(wait, 1.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read URL: {url}")


def yahoo_finance_rss_url(symbol: str) -> str:
    return "https://feeds.finance.yahoo.com/rss/2.0/headline?" + urlencode(
        {"s": symbol, "region": "US", "lang": "en-US"}
    )


def google_news_rss_url(query: str) -> str:
    return "https://news.google.com/rss/search?" + urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )


def fetch_google_news_rss(topic: str, query: str) -> pd.DataFrame:
    payload = _read_url(google_news_rss_url(query), retries=2)
    root = ElementTree.fromstring(payload)
    rows: list[dict[str, object]] = []
    for item in root.findall("./channel/item"):
        published = _parse_time(item.findtext("pubDate"))
        title = (item.findtext("title") or "").strip()
        if published is None or not title:
            continue
        source = item.findtext("source") or "google_news"
        rows.append(
            {
                "published_at": published.isoformat(),
                "symbol": "ALL",
                "headline": title,
                "body": (item.findtext("description") or "").strip(),
                "source": f"google_news_rss:{topic}:{source}",
                "url": (item.findtext("link") or "").strip(),
                "retrieved_at": _utc_now_iso(),
            }
        )
    return pd.DataFrame(rows)


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


def gdelt_doc_url(
    query: str,
    *,
    timespan: str | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    maxrecords: int,
) -> str:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": min(max(int(maxrecords), 1), 250),
    }
    if start_datetime or end_datetime:
        if timespan:
            raise ValueError("GDELT DOC API accepts either timespan or start/end datetime, not both")
        if start_datetime:
            params["startdatetime"] = start_datetime
        if end_datetime:
            params["enddatetime"] = end_datetime
    else:
        params["timespan"] = timespan or "30d"
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)


def fetch_gdelt_articles(
    topic: str,
    query: str,
    *,
    timespan: str = "30d",
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    maxrecords: int = 75,
    sleep_seconds: float = 0.0,
    retries: int = 3,
) -> pd.DataFrame:
    payload = _read_url(
        gdelt_doc_url(
            query,
            timespan=None if start_datetime or end_datetime else timespan,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            maxrecords=maxrecords,
        ),
        sleep_seconds=sleep_seconds,
        retries=retries,
    )
    text = payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GDELT returned non-JSON response: {text[:240]}") from exc
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
    out["published_at"] = pd.to_datetime(out["published_at"], format="mixed", errors="coerce", utc=True)
    out["retrieved_at"] = pd.to_datetime(out["retrieved_at"], format="mixed", errors="coerce", utc=True)
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

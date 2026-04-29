#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oil price forecasting baseline")
    parser.add_argument("--source", choices=["demo", "csv", "yfinance"], default="demo")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--symbol", type=str, default="CL=F")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--target-col", type=str, default="Close")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", choices=["rf", "gbr", "xgb"], default="rf")
    parser.add_argument("--out-dir", type=str, default="outputs")
    return parser.parse_args()


def load_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1400
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    trend = np.linspace(55, 92, n)
    seasonality = 4 * np.sin(np.arange(n) / 28.0) + 2 * np.sin(np.arange(n) / 120.0)
    noise = rng.normal(0, 1.8, n)
    close = trend + seasonality + noise
    volume = rng.integers(80_000, 220_000, size=n)
    return pd.DataFrame({"Date": dates, "Close": close, "Volume": volume})


def load_yfinance_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise ImportError("yfinance is required for --source yfinance") from exc

    frame = yf.download(
        symbol,
        start=start_date,
        end=end_date if end_date else None,
        auto_adjust=True,
        progress=False,
    )
    if frame.empty:
        raise ValueError("No data downloaded from yfinance")
    frame = frame.reset_index()
    frame.columns = flatten_yfinance_columns(frame.columns)
    return frame


def flatten_yfinance_columns(columns: pd.Index) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flattened = []
        for col in columns:
            parts = [str(part) for part in col if str(part) and str(part) != "nan"]
            flattened.append(parts[0] if parts else "")
        return flattened
    return [str(c) for c in columns]


def load_csv_data(csv_path: str) -> pd.DataFrame:
    if not csv_path:
        raise ValueError("--csv-path is required when --source csv")
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(path)


def feature_engineering(df: pd.DataFrame, date_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Date column not found: {date_col}")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).reset_index(drop=True)

    y = frame[target_col].astype(float)
    frame["ret_1"] = y.pct_change()
    frame["lag_1"] = y.shift(1)
    frame["lag_2"] = y.shift(2)
    frame["lag_3"] = y.shift(3)
    frame["lag_7"] = y.shift(7)
    frame["ma_3"] = y.rolling(3).mean()
    frame["ma_7"] = y.rolling(7).mean()
    frame["ma_14"] = y.rolling(14).mean()
    frame["std_7"] = y.rolling(7).std()
    frame["std_14"] = y.rolling(14).std()
    if "Volume" in frame.columns:
        frame["log_volume"] = np.log1p(frame["Volume"].astype(float))

    frame["y"] = y.shift(-horizon)
    frame = frame.dropna().reset_index(drop=True)
    return frame


def build_model(model_name: str):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=1
        )
    if model_name == "gbr":
        return GradientBoostingRegressor(random_state=42)
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is required for --model xgb") from exc
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.source == "demo":
        raw = load_demo_data()
    elif args.source == "yfinance":
        raw = load_yfinance_data(args.symbol, args.start_date, args.end_date)
    else:
        raw = load_csv_data(args.csv_path)

    frame = feature_engineering(raw, args.date_col, args.target_col, args.horizon)
    feature_cols = [c for c in frame.columns if c not in {args.date_col, "y"}]
    X = frame[feature_cols].astype(float)
    y = frame["y"].astype(float).to_numpy()

    split_idx = int(len(frame) * (1 - args.test_size))
    if split_idx < 50:
        raise ValueError("Not enough rows after feature engineering. Use more data.")

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    test_dates = frame[args.date_col].iloc[split_idx:].reset_index(drop=True)

    model = build_model(args.model)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "model": args.model,
        "source": args.source,
        "symbol": args.symbol,
        "horizon": args.horizon,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mape": mape(y_test, pred),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual": y_test,
            "predicted": pred,
            "abs_error": np.abs(y_test - pred),
        }
    )
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    joblib.dump({"model": model, "features": feature_cols}, out_dir / "model.joblib")

    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["date"], pred_df["actual"], label="Actual")
    plt.plot(pred_df["date"], pred_df["predicted"], label="Predicted", alpha=0.8)
    plt.title(f"Oil Price Forecast ({args.model}, horizon={args.horizon})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "forecast_plot.png", dpi=150)
    plt.close()

    return metrics


def main():
    args = parse_args()
    metrics = run_pipeline(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved artifacts to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()

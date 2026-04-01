from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NEW_YORK_TZ = ZoneInfo("America/New_York")


def bars_to_dataframe(bars: Iterable) -> pd.DataFrame:
    rows = []
    for bar in bars:
        timestamp = getattr(bar, "t", None) or getattr(bar, "timestamp", None)
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "open": float(getattr(bar, "o", getattr(bar, "open", 0.0))),
                "high": float(getattr(bar, "h", getattr(bar, "high", 0.0))),
                "low": float(getattr(bar, "l", getattr(bar, "low", 0.0))),
                "close": float(getattr(bar, "c", getattr(bar, "close", 0.0))),
                "volume": float(getattr(bar, "v", getattr(bar, "volume", 0.0))),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = frame.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    return frame


def bars_start_for_timeframe(timeframe: str, limit: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    timeframe = timeframe.strip()

    if timeframe.endswith("Min"):
        minutes_per_bar = max(1, int(timeframe[:-3]))
        bars_per_session = max(1, math.floor(390 / minutes_per_bar))
        trading_days = math.ceil(limit / bars_per_session) + 5
        calendar_days = math.ceil(trading_days * 7 / 5) + 2
    elif timeframe.endswith("Hour"):
        hours_per_bar = max(1, int(timeframe[:-4]))
        bars_per_session = max(1, math.floor(390 / (hours_per_bar * 60)))
        trading_days = math.ceil(limit / bars_per_session) + 5
        calendar_days = math.ceil(trading_days * 7 / 5) + 2
    elif timeframe.endswith("Day"):
        days_per_bar = max(1, int(timeframe[:-3]))
        trading_days = limit * days_per_bar + 20
        calendar_days = math.ceil(trading_days * 7 / 5) + 5
    else:
        calendar_days = max(limit * 3, 30)

    return (now - timedelta(days=calendar_days)).isoformat()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False).mean()


def adx(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )

    average_true_range = atr(frame, length)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / average_true_range
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / average_true_range
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / length, adjust=False).mean().fillna(0.0)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / length, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / length, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("session_vwap expects a DatetimeIndex")

    if frame.index.tz is None:
        localized_index = frame.index.tz_localize("UTC")
    else:
        localized_index = frame.index

    session_labels = localized_index.tz_convert(NEW_YORK_TZ).normalize()
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3
    typical_value = typical_price * frame["volume"]
    cumulative_value = typical_value.groupby(session_labels).cumsum()
    cumulative_volume = frame["volume"].groupby(session_labels).cumsum().replace(0, np.nan)
    return cumulative_value / cumulative_volume


def anchored_vwap(frame: pd.DataFrame, anchor_timestamp: pd.Timestamp | None) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")

    if anchor_timestamp is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    anchor_timestamp = pd.Timestamp(anchor_timestamp)
    anchor_mask = frame.index >= anchor_timestamp
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3
    typical_value = (typical_price * frame["volume"]).where(anchor_mask, 0.0)
    anchored_volume = frame["volume"].where(anchor_mask, 0.0)
    cumulative_value = typical_value.cumsum()
    cumulative_volume = anchored_volume.cumsum().replace(0, np.nan)
    return cumulative_value / cumulative_volume


def supertrend(frame: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    average_true_range = atr(frame, length)
    hl2 = (frame["high"] + frame["low"]) / 2
    upper_band = hl2 + multiplier * average_true_range
    lower_band = hl2 - multiplier * average_true_range

    final_upper_band = upper_band.copy()
    final_lower_band = lower_band.copy()
    direction = pd.Series(index=frame.index, dtype="int64")
    trend_line = pd.Series(index=frame.index, dtype="float64")

    if frame.empty:
        return pd.DataFrame(columns=["supertrend", "supertrend_direction", "upper_band", "lower_band"])

    direction.iloc[0] = 1
    trend_line.iloc[0] = lower_band.iloc[0]

    for index in range(1, len(frame)):
        prev_index = index - 1

        if upper_band.iloc[index] < final_upper_band.iloc[prev_index] or frame["close"].iloc[prev_index] > final_upper_band.iloc[prev_index]:
            final_upper_band.iloc[index] = upper_band.iloc[index]
        else:
            final_upper_band.iloc[index] = final_upper_band.iloc[prev_index]

        if lower_band.iloc[index] > final_lower_band.iloc[prev_index] or frame["close"].iloc[prev_index] < final_lower_band.iloc[prev_index]:
            final_lower_band.iloc[index] = lower_band.iloc[index]
        else:
            final_lower_band.iloc[index] = final_lower_band.iloc[prev_index]

        if frame["close"].iloc[index] > final_upper_band.iloc[prev_index]:
            direction.iloc[index] = 1
        elif frame["close"].iloc[index] < final_lower_band.iloc[prev_index]:
            direction.iloc[index] = -1
        else:
            direction.iloc[index] = direction.iloc[prev_index]

        trend_line.iloc[index] = final_lower_band.iloc[index] if direction.iloc[index] == 1 else final_upper_band.iloc[index]

    return pd.DataFrame(
        {
            "supertrend": trend_line,
            "supertrend_direction": direction.fillna(0).astype(int),
            "upper_band": final_upper_band,
            "lower_band": final_lower_band,
        },
        index=frame.index,
    )


def enrich_ohlcv(
    frame: pd.DataFrame,
    *,
    atr_length: int = 14,
    adx_length: int = 14,
    supertrend_length: int = 10,
    supertrend_multiplier: float = 3.0,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["ema_20"] = ema(enriched["close"], 20)
    enriched["ema_50"] = ema(enriched["close"], 50)
    enriched["ema_200"] = ema(enriched["close"], 200)
    enriched["atr"] = atr(enriched, atr_length)
    enriched["adx"] = adx(enriched, adx_length)
    enriched["rsi"] = rsi(enriched["close"], 14)
    enriched["vwap"] = session_vwap(enriched)
    enriched["avg_volume_20"] = enriched["volume"].rolling(20).mean()
    enriched["avg_dollar_volume_20"] = (enriched["close"] * enriched["volume"]).rolling(20).mean()
    supertrend_frame = supertrend(enriched, supertrend_length, supertrend_multiplier)
    for column in supertrend_frame.columns:
        enriched[column] = supertrend_frame[column]
    return enriched
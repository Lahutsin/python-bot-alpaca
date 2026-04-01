from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from indicators import anchored_vwap, enrich_ohlcv


@dataclass
class StrategyConfig:
    market_symbol: str = "SPY"
    benchmark_symbol: str = "QQQ"
    volatility_symbol: str | None = None
    volatility_threshold: float = 35.0
    market_regime_ema_buffer_pct: float = 0.10
    stock_trend_ema_buffer_pct: float = 0.10
    daily_timeframe: str = "1Day"
    pullback_timeframe: str = "1Hour"
    entry_timeframe: str = "15Min"
    risk_per_trade_pct: float = 0.5
    min_risk_per_trade_pct: float = 0.15
    max_position_pct: float = 10.0
    max_open_positions: int = 3
    max_correlated_positions: int = 2
    correlation_threshold: float = 0.8
    correlation_lookback: int = 60
    daily_loss_limit_pct: float = 2.0
    weekly_loss_limit_pct: float = 5.0
    drawdown_risk_scale_trigger_pct: float = 3.0
    drawdown_risk_multiplier: float = 0.5
    losing_streak_risk_multiplier: float = 0.8
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 180
    supertrend_length: int = 10
    supertrend_multiplier: float = 3.0
    require_daily_supertrend: bool = True
    atr_length: int = 14
    atr_stop_multiplier: float = 1.5
    atr_trailing_multiplier: float = 2.0
    adx_length: int = 14
    adx_threshold: float = 20.0
    min_avg_daily_dollar_volume: float = 2_000_000.0
    min_spread_pct: float = 0.30
    max_spread_pct: float = 2.00
    spread_atr_ratio: float = 0.50
    volume_multiplier: float = 1.2
    max_extension_atr: float = 2.5
    max_entry_rsi: float = 68.0
    pullback_tolerance_pct: float = 0.5
    partial_take_profit_r: float = 1.0
    partial_exit_pct: float = 0.5
    relative_strength_lookback: int = 60
    min_relative_strength: float = 0.0
    min_entry_score: float = 5.5
    breakout_buffer_pct: float = 0.0
    breakout_lookback_bars: int = 4
    entry_vwap_buffer_pct: float = 0.0
    anchored_vwap_lookback_bars: int = 10
    require_pullback_above_ema50: bool = False
    require_entry_above_ema20: bool = False
    require_entry_supertrend: bool = False
    require_entry_above_vwap: bool = True
    require_rising_entry_vwap: bool = True
    require_entry_above_anchored_vwap: bool = True
    require_rising_anchored_vwap: bool = False
    daily_exit_confirm_bars: int = 2
    intraday_exit_confirm_bars: int = 2
    break_even_trigger_r: float = 0.75

    @classmethod
    def from_module(cls, module: Any) -> "StrategyConfig":
        return cls(
            market_symbol=getattr(module, "ALPACA_MARKET_SYMBOL", cls.market_symbol),
            benchmark_symbol=getattr(module, "ALPACA_BENCHMARK_SYMBOL", cls.benchmark_symbol),
            volatility_symbol=getattr(module, "ALPACA_VOLATILITY_SYMBOL", cls.volatility_symbol),
            volatility_threshold=float(getattr(module, "ALPACA_VOLATILITY_THRESHOLD", cls.volatility_threshold)),
            market_regime_ema_buffer_pct=float(getattr(module, "ALPACA_MARKET_REGIME_EMA_BUFFER_PCT", cls.market_regime_ema_buffer_pct)),
            stock_trend_ema_buffer_pct=float(getattr(module, "ALPACA_STOCK_TREND_EMA_BUFFER_PCT", cls.stock_trend_ema_buffer_pct)),
            daily_timeframe=getattr(module, "ALPACA_DAILY_TIMEFRAME", cls.daily_timeframe),
            pullback_timeframe=getattr(module, "ALPACA_PULLBACK_TIMEFRAME", cls.pullback_timeframe),
            entry_timeframe=getattr(module, "ALPACA_ENTRY_TIMEFRAME", cls.entry_timeframe),
            risk_per_trade_pct=float(getattr(module, "ALPACA_RISK_PER_TRADE_PCT", cls.risk_per_trade_pct)),
            min_risk_per_trade_pct=float(getattr(module, "ALPACA_MIN_RISK_PER_TRADE_PCT", cls.min_risk_per_trade_pct)),
            max_position_pct=float(getattr(module, "ALPACA_MAX_POSITION_PCT", cls.max_position_pct)),
            max_open_positions=int(getattr(module, "ALPACA_MAX_OPEN_POSITIONS", cls.max_open_positions)),
            max_correlated_positions=int(getattr(module, "ALPACA_MAX_CORRELATED_POSITIONS", cls.max_correlated_positions)),
            correlation_threshold=float(getattr(module, "ALPACA_CORRELATION_THRESHOLD", cls.correlation_threshold)),
            correlation_lookback=int(getattr(module, "ALPACA_CORRELATION_LOOKBACK", cls.correlation_lookback)),
            daily_loss_limit_pct=float(getattr(module, "ALPACA_DAILY_LOSS_LIMIT_PCT", cls.daily_loss_limit_pct)),
            weekly_loss_limit_pct=float(getattr(module, "ALPACA_WEEKLY_LOSS_LIMIT_PCT", cls.weekly_loss_limit_pct)),
            drawdown_risk_scale_trigger_pct=float(getattr(module, "ALPACA_DRAWDOWN_RISK_SCALE_TRIGGER_PCT", cls.drawdown_risk_scale_trigger_pct)),
            drawdown_risk_multiplier=float(getattr(module, "ALPACA_DRAWDOWN_RISK_MULTIPLIER", cls.drawdown_risk_multiplier)),
            losing_streak_risk_multiplier=float(getattr(module, "ALPACA_LOSING_STREAK_RISK_MULTIPLIER", cls.losing_streak_risk_multiplier)),
            max_consecutive_losses=int(getattr(module, "ALPACA_MAX_CONSECUTIVE_LOSSES", cls.max_consecutive_losses)),
            cooldown_minutes=int(getattr(module, "ALPACA_COOLDOWN_MINUTES", cls.cooldown_minutes)),
            supertrend_length=int(getattr(module, "ALPACA_SUPERTREND_LENGTH", cls.supertrend_length)),
            supertrend_multiplier=float(getattr(module, "ALPACA_SUPERTREND_MULTIPLIER", cls.supertrend_multiplier)),
            require_daily_supertrend=bool(getattr(module, "ALPACA_REQUIRE_DAILY_SUPERTREND", cls.require_daily_supertrend)),
            atr_length=int(getattr(module, "ALPACA_ATR_LENGTH", cls.atr_length)),
            atr_stop_multiplier=float(getattr(module, "ALPACA_ATR_STOP_MULTIPLIER", cls.atr_stop_multiplier)),
            atr_trailing_multiplier=float(getattr(module, "ALPACA_ATR_TRAILING_MULTIPLIER", cls.atr_trailing_multiplier)),
            adx_length=int(getattr(module, "ALPACA_ADX_LENGTH", cls.adx_length)),
            adx_threshold=float(getattr(module, "ALPACA_ADX_THRESHOLD", cls.adx_threshold)),
            min_avg_daily_dollar_volume=float(getattr(module, "ALPACA_MIN_AVG_DOLLAR_VOLUME", cls.min_avg_daily_dollar_volume)),
            min_spread_pct=float(getattr(module, "ALPACA_MIN_SPREAD_PCT", cls.min_spread_pct)),
            max_spread_pct=float(getattr(module, "ALPACA_MAX_SPREAD_PCT", cls.max_spread_pct)),
            spread_atr_ratio=float(getattr(module, "ALPACA_SPREAD_ATR_RATIO", cls.spread_atr_ratio)),
            volume_multiplier=float(getattr(module, "ALPACA_ENTRY_VOLUME_MULTIPLIER", cls.volume_multiplier)),
            max_extension_atr=float(getattr(module, "ALPACA_MAX_EXTENSION_ATR", cls.max_extension_atr)),
            max_entry_rsi=float(getattr(module, "ALPACA_MAX_ENTRY_RSI", cls.max_entry_rsi)),
            pullback_tolerance_pct=float(getattr(module, "ALPACA_PULLBACK_TOLERANCE_PCT", cls.pullback_tolerance_pct)),
            partial_take_profit_r=float(getattr(module, "ALPACA_PARTIAL_TAKE_PROFIT_R", cls.partial_take_profit_r)),
            partial_exit_pct=float(getattr(module, "ALPACA_PARTIAL_EXIT_PCT", cls.partial_exit_pct)),
            relative_strength_lookback=int(getattr(module, "ALPACA_RELATIVE_STRENGTH_LOOKBACK", cls.relative_strength_lookback)),
            min_relative_strength=float(getattr(module, "ALPACA_MIN_RELATIVE_STRENGTH", cls.min_relative_strength)),
            min_entry_score=float(getattr(module, "ALPACA_MIN_ENTRY_SCORE", cls.min_entry_score)),
            breakout_buffer_pct=float(getattr(module, "ALPACA_BREAKOUT_BUFFER_PCT", cls.breakout_buffer_pct)),
            breakout_lookback_bars=int(getattr(module, "ALPACA_BREAKOUT_LOOKBACK_BARS", cls.breakout_lookback_bars)),
            entry_vwap_buffer_pct=float(getattr(module, "ALPACA_ENTRY_VWAP_BUFFER_PCT", cls.entry_vwap_buffer_pct)),
            anchored_vwap_lookback_bars=int(getattr(module, "ALPACA_ANCHORED_VWAP_LOOKBACK_BARS", cls.anchored_vwap_lookback_bars)),
            require_pullback_above_ema50=bool(getattr(module, "ALPACA_REQUIRE_PULLBACK_ABOVE_EMA50", cls.require_pullback_above_ema50)),
            require_entry_above_ema20=bool(getattr(module, "ALPACA_REQUIRE_ENTRY_ABOVE_EMA20", cls.require_entry_above_ema20)),
            require_entry_supertrend=bool(getattr(module, "ALPACA_REQUIRE_ENTRY_SUPERTREND", cls.require_entry_supertrend)),
            require_entry_above_vwap=bool(getattr(module, "ALPACA_REQUIRE_ENTRY_ABOVE_VWAP", cls.require_entry_above_vwap)),
            require_rising_entry_vwap=bool(getattr(module, "ALPACA_REQUIRE_RISING_ENTRY_VWAP", cls.require_rising_entry_vwap)),
            require_entry_above_anchored_vwap=bool(getattr(module, "ALPACA_REQUIRE_ENTRY_ABOVE_ANCHORED_VWAP", cls.require_entry_above_anchored_vwap)),
            require_rising_anchored_vwap=bool(getattr(module, "ALPACA_REQUIRE_RISING_ANCHORED_VWAP", cls.require_rising_anchored_vwap)),
            daily_exit_confirm_bars=int(getattr(module, "ALPACA_DAILY_EXIT_CONFIRM_BARS", cls.daily_exit_confirm_bars)),
            intraday_exit_confirm_bars=int(getattr(module, "ALPACA_INTRADAY_EXIT_CONFIRM_BARS", cls.intraday_exit_confirm_bars)),
            break_even_trigger_r=float(getattr(module, "ALPACA_BREAK_EVEN_TRIGGER_R", cls.break_even_trigger_r)),
        )


@dataclass
class SignalDecision:
    action: str
    reason: str
    entry_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    trailing_stop_price: float | None = None
    atr_value: float | None = None
    risk_per_share: float | None = None
    spread_pct: float | None = None
    signal_score: float = 0.0
    relative_strength: float | None = None
    anchored_vwap: float | None = None
    anchored_vwap_anchor_time: str | None = None
    market_regime_reason: str | None = None


class StrategyEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config

    @staticmethod
    def _fmt(value: Any) -> str:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.2f}"

    def allowed_spread_pct(self, daily_row: pd.Series) -> float:
        atr_value = daily_row.get("atr")
        close_price = daily_row.get("close")
        if pd.isna(atr_value) or pd.isna(close_price) or float(close_price) <= 0:
            return self.config.max_spread_pct

        atr_pct = (float(atr_value) / float(close_price)) * 100
        adaptive_spread = atr_pct * self.config.spread_atr_ratio
        return min(self.config.max_spread_pct, max(self.config.min_spread_pct, adaptive_spread))

    def prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return enrich_ohlcv(
            frame,
            atr_length=self.config.atr_length,
            adx_length=self.config.adx_length,
            supertrend_length=self.config.supertrend_length,
            supertrend_multiplier=self.config.supertrend_multiplier,
        )

    def evaluate_market_regime(
        self,
        market_daily: pd.DataFrame,
        benchmark_daily: pd.DataFrame,
        volatility_daily: pd.DataFrame | None = None,
    ) -> tuple[bool, str]:
        if market_daily.empty or benchmark_daily.empty:
            return False, "market data unavailable"

        market_row = market_daily.iloc[-1]
        benchmark_row = benchmark_daily.iloc[-1]
        buffer_multiplier = 1 - (self.config.market_regime_ema_buffer_pct / 100)
        market_threshold = float(market_row["ema_200"]) * buffer_multiplier
        benchmark_threshold = float(benchmark_row["ema_200"]) * buffer_multiplier

        if float(market_row["close"]) < market_threshold:
            return (
                False,
                f"{self.config.market_symbol} below EMA200 buffer: close={self._fmt(market_row['close'])}, ema200={self._fmt(market_row['ema_200'])}, threshold={market_threshold:.2f}, buffer_pct={self.config.market_regime_ema_buffer_pct:.2f}",
            )
        if float(benchmark_row["close"]) < benchmark_threshold:
            return (
                False,
                f"{self.config.benchmark_symbol} below EMA200 buffer: close={self._fmt(benchmark_row['close'])}, ema200={self._fmt(benchmark_row['ema_200'])}, threshold={benchmark_threshold:.2f}, buffer_pct={self.config.market_regime_ema_buffer_pct:.2f}",
            )

        if volatility_daily is not None and not volatility_daily.empty:
            volatility_close = float(volatility_daily.iloc[-1]["close"])
            if volatility_close >= self.config.volatility_threshold:
                return (
                    False,
                    f"volatility proxy too high: close={volatility_close:.2f}, threshold={self.config.volatility_threshold:.2f}",
                )

        return True, "market regime is long-biased"

    def calculate_effective_risk_pct(self, current_equity: float, reference_equity: float, consecutive_losses: int) -> float:
        risk_pct = self.config.risk_per_trade_pct
        if consecutive_losses > 0:
            risk_pct *= self.config.losing_streak_risk_multiplier ** consecutive_losses

        if reference_equity > 0:
            drawdown_pct = ((current_equity - reference_equity) / reference_equity) * 100
            if drawdown_pct <= -self.config.drawdown_risk_scale_trigger_pct:
                risk_pct *= self.config.drawdown_risk_multiplier

        return max(risk_pct, self.config.min_risk_per_trade_pct)

    def compute_relative_strength(self, symbol_daily: pd.DataFrame, benchmark_daily: pd.DataFrame | None) -> float | None:
        if benchmark_daily is None or benchmark_daily.empty:
            return None

        lookback = self.config.relative_strength_lookback
        aligned = pd.concat(
            [symbol_daily[["close"]].rename(columns={"close": "symbol"}), benchmark_daily[["close"]].rename(columns={"close": "benchmark"})],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) <= lookback:
            return None

        symbol_return = aligned["symbol"].iloc[-1] / aligned["symbol"].iloc[-lookback - 1] - 1
        benchmark_return = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-lookback - 1] - 1
        return float(symbol_return - benchmark_return)

    def pullback_anchor_timestamp(self, pullback_frame: pd.DataFrame) -> pd.Timestamp | None:
        if pullback_frame.empty:
            return None

        lookback = min(len(pullback_frame), max(2, self.config.anchored_vwap_lookback_bars))
        anchor_window = pullback_frame.iloc[-lookback:]
        if anchor_window.empty:
            return None

        return pd.Timestamp(anchor_window["low"].idxmin())

    def compute_entry_anchored_vwap(self, pullback_frame: pd.DataFrame, entry_frame: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Series]:
        anchor_timestamp = self.pullback_anchor_timestamp(pullback_frame)
        if anchor_timestamp is None:
            return None, pd.Series(dtype="float64")
        return anchor_timestamp, anchored_vwap(entry_frame, anchor_timestamp)

    def score_entry_setup(
        self,
        daily_row: pd.Series,
        pullback_row: pd.Series,
        entry_row: pd.Series,
        extension_atr: float,
        volume_ratio: float,
        relative_strength: float | None,
    ) -> float:
        score = 2.0
        score += min(max((float(daily_row["adx"]) - self.config.adx_threshold) / 10, 0.0), 2.0)
        score += min(max((volume_ratio - self.config.volume_multiplier) * 2, 0.0), 2.0)
        score += min(max((self.config.max_extension_atr - extension_atr) / max(self.config.max_extension_atr, 1), 0.0), 1.0)
        score += min(max((self.config.max_entry_rsi - float(daily_row["rsi"])) / 12, 0.0), 1.0)
        score += 1.0 if float(entry_row["close"]) > float(entry_row["ema_20"]) else 0.0
        score += 1.0 if float(pullback_row["close"]) >= float(pullback_row["ema_20"]) else 0.5

        if relative_strength is not None:
            score += min(max(relative_strength / 0.03, 0.0), 2.0)
        else:
            score += 0.5

        return round(score, 2)

    def build_entry_decision(
        self,
        symbol: str,
        daily_frame: pd.DataFrame,
        pullback_frame: pd.DataFrame,
        entry_frame: pd.DataFrame,
        current_price: float,
        spread_pct: float | None,
        market_regime: tuple[bool, str],
        benchmark_daily: pd.DataFrame | None = None,
    ) -> SignalDecision:
        regime_ok, regime_reason = market_regime
        if not regime_ok:
            return SignalDecision("hold", regime_reason, market_regime_reason=regime_reason)

        if any(frame.empty for frame in (daily_frame, pullback_frame, entry_frame)):
            return SignalDecision(
                "hold",
                f"{symbol}: incomplete OHLCV history: daily={len(daily_frame)}, pullback={len(pullback_frame)}, entry={len(entry_frame)}",
                market_regime_reason=regime_reason,
            )

        daily_row = daily_frame.iloc[-1]
        pullback_row = pullback_frame.iloc[-1]
        entry_row = entry_frame.iloc[-1]
        previous_entry_row = entry_frame.iloc[-2] if len(entry_frame) > 1 else None
        relative_strength = self.compute_relative_strength(daily_frame, benchmark_daily)
        stock_buffer_multiplier = 1 - (self.config.stock_trend_ema_buffer_pct / 100)
        stock_ema_threshold = float(daily_row["ema_200"]) * stock_buffer_multiplier
        ema50_threshold = float(daily_row["ema_200"]) * stock_buffer_multiplier
        allowed_spread_pct = self.allowed_spread_pct(daily_row)

        if spread_pct is not None and spread_pct > allowed_spread_pct:
            return SignalDecision(
                "hold",
                f"{symbol}: spread too wide: spread={spread_pct:.2f}%, allowed={allowed_spread_pct:.2f}%, atr_pct={(float(daily_row['atr']) / float(daily_row['close']) * 100) if float(daily_row['close']) else 0.0:.2f}%",
                spread_pct=spread_pct,
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if daily_row["avg_dollar_volume_20"] < self.config.min_avg_daily_dollar_volume:
            return SignalDecision(
                "hold",
                f"{symbol}: dollar volume too low: avg_dollar_volume_20={self._fmt(daily_row['avg_dollar_volume_20'])}, min={self.config.min_avg_daily_dollar_volume:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if float(daily_row["close"]) < stock_ema_threshold:
            return SignalDecision(
                "hold",
                f"{symbol}: close below EMA200 buffer: close={self._fmt(daily_row['close'])}, ema200={self._fmt(daily_row['ema_200'])}, threshold={stock_ema_threshold:.2f}, buffer_pct={self.config.stock_trend_ema_buffer_pct:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if float(daily_row["ema_50"]) < ema50_threshold:
            return SignalDecision(
                "hold",
                f"{symbol}: EMA50 below EMA200 buffer: ema50={self._fmt(daily_row['ema_50'])}, ema200={self._fmt(daily_row['ema_200'])}, threshold={ema50_threshold:.2f}, buffer_pct={self.config.stock_trend_ema_buffer_pct:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if self.config.require_daily_supertrend and daily_row["supertrend_direction"] != 1:
            return SignalDecision(
                "hold",
                f"{symbol}: daily supertrend not long: direction={int(daily_row['supertrend_direction'])}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if daily_row["adx"] < self.config.adx_threshold:
            return SignalDecision(
                "hold",
                f"{symbol}: ADX below threshold: adx={self._fmt(daily_row['adx'])}, min={self.config.adx_threshold:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if pd.isna(daily_row["atr"]) or daily_row["atr"] <= 0:
            return SignalDecision(
                "hold",
                f"{symbol}: ATR unavailable: atr={self._fmt(daily_row['atr'])}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        extension_atr = (daily_row["close"] - daily_row["ema_20"]) / daily_row["atr"]
        if extension_atr > self.config.max_extension_atr:
            return SignalDecision(
                "hold",
                f"{symbol}: price too extended from EMA20: extension_atr={float(extension_atr):.2f}, max={self.config.max_extension_atr:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if daily_row["rsi"] > self.config.max_entry_rsi:
            return SignalDecision(
                "hold",
                f"{symbol}: RSI too high: rsi={self._fmt(daily_row['rsi'])}, max={self.config.max_entry_rsi:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if relative_strength is not None and relative_strength < self.config.min_relative_strength:
            return SignalDecision(
                "hold",
                f"{symbol}: relative strength too weak: rs={relative_strength:.4f}, min={self.config.min_relative_strength:.4f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        tolerance = self.config.pullback_tolerance_pct / 100
        touched_pullback_zone = (
            pullback_row["low"] <= pullback_row["ema_20"] * (1 + tolerance)
            or pullback_row["low"] <= pullback_row["ema_50"] * (1 + tolerance)
        )
        if not touched_pullback_zone:
            return SignalDecision(
                "hold",
                f"{symbol}: no pullback on {self.config.pullback_timeframe}: low={self._fmt(pullback_row['low'])}, ema20={self._fmt(pullback_row['ema_20'])}, ema50={self._fmt(pullback_row['ema_50'])}, tolerance_pct={self.config.pullback_tolerance_pct:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if self.config.require_pullback_above_ema50 and float(pullback_row["close"]) < float(pullback_row["ema_50"]):
            return SignalDecision(
                "hold",
                f"{symbol}: pullback close below EMA50: close={self._fmt(pullback_row['close'])}, ema50={self._fmt(pullback_row['ema_50'])}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        breakout_lookback = max(1, self.config.breakout_lookback_bars)
        if previous_entry_row is None or len(entry_frame) <= breakout_lookback:
            return SignalDecision(
                "hold",
                f"{symbol}: not enough entry bars: entry_bars={len(entry_frame)}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        breakout_reference = float(entry_frame.iloc[-breakout_lookback - 1 : -1]["high"].max())
        breakout_threshold = breakout_reference * (1 + self.config.breakout_buffer_pct / 100)
        breakout = float(entry_row["close"]) >= breakout_threshold
        volume_ratio = float(entry_row["volume"] / entry_row["avg_volume_20"]) if entry_row["avg_volume_20"] and entry_row["avg_volume_20"] > 0 else 0.0
        volume_ok = volume_ratio >= self.config.volume_multiplier
        entry_vwap = entry_row.get("vwap")
        previous_entry_vwap = previous_entry_row.get("vwap") if previous_entry_row is not None else pd.NA
        vwap_threshold = float(entry_vwap) * (1 + self.config.entry_vwap_buffer_pct / 100) if not pd.isna(entry_vwap) else None
        anchored_vwap_anchor_time, anchored_vwap_series = self.compute_entry_anchored_vwap(pullback_frame, entry_frame)
        entry_anchored_vwap = anchored_vwap_series.iloc[-1] if not anchored_vwap_series.empty else pd.NA
        previous_entry_anchored_vwap = anchored_vwap_series.iloc[-2] if len(anchored_vwap_series) > 1 else pd.NA

        if self.config.require_entry_above_ema20 and float(entry_row["close"]) < float(entry_row["ema_20"]):
            return SignalDecision(
                "hold",
                f"{symbol}: entry close below EMA20: close={self._fmt(entry_row['close'])}, ema20={self._fmt(entry_row['ema_20'])}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if self.config.require_entry_supertrend and int(entry_row["supertrend_direction"]) != 1:
            return SignalDecision(
                "hold",
                f"{symbol}: entry supertrend not long: direction={int(entry_row['supertrend_direction'])}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if not breakout:
            return SignalDecision(
                "hold",
                f"{symbol}: no breakout on {self.config.entry_timeframe}: close={self._fmt(entry_row['close'])}, threshold={breakout_threshold:.2f}, lookback_high={breakout_reference:.2f}, lookback_bars={breakout_lookback}, buffer_pct={self.config.breakout_buffer_pct:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )
        if not volume_ok:
            return SignalDecision(
                "hold",
                f"{symbol}: breakout volume too weak: volume_ratio={volume_ratio:.2f}, min={self.config.volume_multiplier:.2f}",
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        if self.config.require_entry_above_vwap:
            if vwap_threshold is None:
                return SignalDecision(
                    "hold",
                    f"{symbol}: VWAP unavailable on {self.config.entry_timeframe}",
                    relative_strength=relative_strength,
                    market_regime_reason=regime_reason,
                )
            if float(entry_row["close"]) < vwap_threshold:
                return SignalDecision(
                    "hold",
                    f"{symbol}: entry close below VWAP: close={self._fmt(entry_row['close'])}, vwap={self._fmt(entry_vwap)}, threshold={vwap_threshold:.2f}, buffer_pct={self.config.entry_vwap_buffer_pct:.2f}",
                    relative_strength=relative_strength,
                    market_regime_reason=regime_reason,
                )

        if self.config.require_rising_entry_vwap:
            if pd.isna(entry_vwap) or pd.isna(previous_entry_vwap):
                return SignalDecision(
                    "hold",
                    f"{symbol}: insufficient VWAP history on {self.config.entry_timeframe}",
                    relative_strength=relative_strength,
                    market_regime_reason=regime_reason,
                )
            if float(entry_vwap) <= float(previous_entry_vwap):
                return SignalDecision(
                    "hold",
                    f"{symbol}: VWAP not rising: current={self._fmt(entry_vwap)}, previous={self._fmt(previous_entry_vwap)}",
                    relative_strength=relative_strength,
                    market_regime_reason=regime_reason,
                )

        if self.config.require_entry_above_anchored_vwap:
            if pd.isna(entry_anchored_vwap):
                return SignalDecision(
                    "hold",
                    f"{symbol}: Anchored VWAP unavailable on {self.config.entry_timeframe}",
                    relative_strength=relative_strength,
                    market_regime_reason=regime_reason,
                )
            if float(entry_row["close"]) < float(entry_anchored_vwap):
                return SignalDecision(
                    "hold",
                    f"{symbol}: entry close below Anchored VWAP: close={self._fmt(entry_row['close'])}, anchored_vwap={self._fmt(entry_anchored_vwap)}",
                    relative_strength=relative_strength,
                    anchored_vwap=float(entry_anchored_vwap),
                    anchored_vwap_anchor_time=None if anchored_vwap_anchor_time is None else str(anchored_vwap_anchor_time),
                    market_regime_reason=regime_reason,
                )

        if self.config.require_rising_anchored_vwap:
            if pd.isna(entry_anchored_vwap) or pd.isna(previous_entry_anchored_vwap):
                return SignalDecision(
                    "hold",
                    f"{symbol}: insufficient Anchored VWAP history on {self.config.entry_timeframe}",
                    relative_strength=relative_strength,
                    anchored_vwap=float(entry_anchored_vwap) if not pd.isna(entry_anchored_vwap) else None,
                    anchored_vwap_anchor_time=None if anchored_vwap_anchor_time is None else str(anchored_vwap_anchor_time),
                    market_regime_reason=regime_reason,
                )
            if float(entry_anchored_vwap) <= float(previous_entry_anchored_vwap):
                return SignalDecision(
                    "hold",
                    f"{symbol}: Anchored VWAP not rising: current={self._fmt(entry_anchored_vwap)}, previous={self._fmt(previous_entry_anchored_vwap)}",
                    relative_strength=relative_strength,
                    anchored_vwap=float(entry_anchored_vwap),
                    anchored_vwap_anchor_time=None if anchored_vwap_anchor_time is None else str(anchored_vwap_anchor_time),
                    market_regime_reason=regime_reason,
                )

        signal_score = self.score_entry_setup(daily_row, pullback_row, entry_row, float(extension_atr), volume_ratio, relative_strength)
        if signal_score < self.config.min_entry_score:
            return SignalDecision(
                "hold",
                f"{symbol}: setup score below threshold: score={signal_score:.2f}, min={self.config.min_entry_score:.2f}",
                signal_score=signal_score,
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        atr_value = float(daily_row["atr"])
        stop_buffer = atr_value * self.config.atr_stop_multiplier
        stop_candidates = [
            current_price - stop_buffer,
            float(pullback_row["low"] - atr_value * 0.15),
            float(entry_row["low"] - atr_value * 0.10),
        ]
        stop_price = min(stop_candidates)
        risk_per_share = current_price - stop_price
        if risk_per_share <= 0:
            return SignalDecision(
                "hold",
                f"{symbol}: invalid stop distance: entry={current_price:.2f}, stop={stop_price:.2f}, risk_per_share={risk_per_share:.4f}",
                signal_score=signal_score,
                relative_strength=relative_strength,
                market_regime_reason=regime_reason,
            )

        take_profit_price = current_price + risk_per_share * self.config.partial_take_profit_r
        return SignalDecision(
            action="buy",
            reason=f"{symbol}: ranked setup score {signal_score:.2f}",
            entry_price=current_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            trailing_stop_price=stop_price,
            atr_value=atr_value,
            risk_per_share=risk_per_share,
            spread_pct=spread_pct,
            signal_score=signal_score,
            relative_strength=relative_strength,
            anchored_vwap=float(entry_anchored_vwap) if not pd.isna(entry_anchored_vwap) else None,
            anchored_vwap_anchor_time=None if anchored_vwap_anchor_time is None else str(anchored_vwap_anchor_time),
            market_regime_reason=regime_reason,
        )

    def build_exit_decision(
        self,
        symbol: str,
        trade_state: dict[str, Any],
        daily_frame: pd.DataFrame,
        pullback_frame: pd.DataFrame,
        current_price: float,
        market_regime: tuple[bool, str],
    ) -> SignalDecision:
        if daily_frame.empty or pullback_frame.empty:
            return SignalDecision("hold", f"{symbol}: incomplete data for exit")

        daily_row = daily_frame.iloc[-1]
        pullback_row = pullback_frame.iloc[-1]
        daily_confirm_bars = max(1, self.config.daily_exit_confirm_bars)
        previous_daily_rows = daily_frame.iloc[-daily_confirm_bars:] if len(daily_frame) >= daily_confirm_bars else daily_frame
        previous_pullback_rows = pullback_frame.iloc[-self.config.intraday_exit_confirm_bars :] if len(pullback_frame) >= self.config.intraday_exit_confirm_bars else pullback_frame

        stored_stop = float(trade_state["stop_price"])
        atr_value = float(daily_row["atr"]) if not pd.isna(daily_row["atr"]) else trade_state.get("atr_value", 0.0)
        max_price = max(float(trade_state.get("max_price", current_price)), current_price)
        entry_price = float(trade_state.get("entry_price", current_price))
        risk_per_share = max(float(trade_state.get("risk_per_share", 0.0)), 0.0)
        trade_in_profit = current_price >= entry_price
        trailing_stop = max(
            stored_stop,
            max_price - atr_value * self.config.atr_trailing_multiplier if atr_value > 0 else stored_stop,
            float(daily_row["supertrend"]) if daily_row["supertrend_direction"] == 1 else stored_stop,
        )

        if risk_per_share > 0 and max_price >= entry_price + risk_per_share * self.config.break_even_trigger_r:
            trailing_stop = max(trailing_stop, entry_price)

        regime_ok, regime_reason = market_regime
        if current_price <= trailing_stop:
            stop_reason = "hard stop hit" if trailing_stop <= stored_stop else "trailing stop hit"
            return SignalDecision("sell", f"{symbol}: {stop_reason}", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if not regime_ok:
            return SignalDecision("sell", f"{symbol}: {regime_reason}", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if self.config.require_daily_supertrend and daily_row["supertrend_direction"] != 1:
            return SignalDecision("sell", f"{symbol}: daily supertrend flipped", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if len(previous_daily_rows) >= daily_confirm_bars:
            closes_below_daily_ema50 = all(float(row["close"]) < float(row["ema_50"]) for _, row in previous_daily_rows.iterrows())
            if closes_below_daily_ema50 and (trade_state.get("partial_exit_done") or trade_in_profit):
                return SignalDecision("sell", f"{symbol}: confirmed daily close below EMA50", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        confirm_bars = max(1, self.config.intraday_exit_confirm_bars)
        if len(previous_pullback_rows) >= confirm_bars:
            closes_below_ema50 = all(float(row["close"]) < float(row["ema_50"]) for _, row in previous_pullback_rows.iterrows())
            if closes_below_ema50 and int(pullback_row["supertrend_direction"]) != 1 and (trade_state.get("partial_exit_done") or trade_in_profit):
                return SignalDecision("sell", f"{symbol}: confirmed intraday trend break", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if not trade_state.get("partial_exit_done") and current_price >= float(trade_state["take_profit_price"]):
            return SignalDecision("partial_sell", f"{symbol}: first target reached", stop_price=stored_stop, trailing_stop_price=max(trailing_stop, float(trade_state["entry_price"])), atr_value=atr_value, market_regime_reason=regime_reason)

        return SignalDecision("hold", f"{symbol}: keep position", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)
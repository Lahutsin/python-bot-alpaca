from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from indicators import enrich_ohlcv


@dataclass
class StrategyConfig:
    market_symbol: str = "SPY"
    benchmark_symbol: str = "QQQ"
    volatility_symbol: str | None = None
    volatility_threshold: float = 35.0
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
    atr_length: int = 14
    atr_stop_multiplier: float = 1.5
    atr_trailing_multiplier: float = 2.0
    adx_length: int = 14
    adx_threshold: float = 20.0
    min_avg_daily_dollar_volume: float = 2_000_000.0
    max_spread_pct: float = 0.30
    volume_multiplier: float = 1.2
    max_extension_atr: float = 2.5
    max_entry_rsi: float = 68.0
    pullback_tolerance_pct: float = 0.5
    partial_take_profit_r: float = 1.0
    partial_exit_pct: float = 0.5
    relative_strength_lookback: int = 60
    min_relative_strength: float = 0.0
    min_entry_score: float = 5.5

    @classmethod
    def from_module(cls, module: Any) -> "StrategyConfig":
        return cls(
            market_symbol=getattr(module, "ALPACA_MARKET_SYMBOL", cls.market_symbol),
            benchmark_symbol=getattr(module, "ALPACA_BENCHMARK_SYMBOL", cls.benchmark_symbol),
            volatility_symbol=getattr(module, "ALPACA_VOLATILITY_SYMBOL", cls.volatility_symbol),
            volatility_threshold=float(getattr(module, "ALPACA_VOLATILITY_THRESHOLD", cls.volatility_threshold)),
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
            atr_length=int(getattr(module, "ALPACA_ATR_LENGTH", cls.atr_length)),
            atr_stop_multiplier=float(getattr(module, "ALPACA_ATR_STOP_MULTIPLIER", cls.atr_stop_multiplier)),
            atr_trailing_multiplier=float(getattr(module, "ALPACA_ATR_TRAILING_MULTIPLIER", cls.atr_trailing_multiplier)),
            adx_length=int(getattr(module, "ALPACA_ADX_LENGTH", cls.adx_length)),
            adx_threshold=float(getattr(module, "ALPACA_ADX_THRESHOLD", cls.adx_threshold)),
            min_avg_daily_dollar_volume=float(getattr(module, "ALPACA_MIN_AVG_DOLLAR_VOLUME", cls.min_avg_daily_dollar_volume)),
            max_spread_pct=float(getattr(module, "ALPACA_MAX_SPREAD_PCT", cls.max_spread_pct)),
            volume_multiplier=float(getattr(module, "ALPACA_ENTRY_VOLUME_MULTIPLIER", cls.volume_multiplier)),
            max_extension_atr=float(getattr(module, "ALPACA_MAX_EXTENSION_ATR", cls.max_extension_atr)),
            max_entry_rsi=float(getattr(module, "ALPACA_MAX_ENTRY_RSI", cls.max_entry_rsi)),
            pullback_tolerance_pct=float(getattr(module, "ALPACA_PULLBACK_TOLERANCE_PCT", cls.pullback_tolerance_pct)),
            partial_take_profit_r=float(getattr(module, "ALPACA_PARTIAL_TAKE_PROFIT_R", cls.partial_take_profit_r)),
            partial_exit_pct=float(getattr(module, "ALPACA_PARTIAL_EXIT_PCT", cls.partial_exit_pct)),
            relative_strength_lookback=int(getattr(module, "ALPACA_RELATIVE_STRENGTH_LOOKBACK", cls.relative_strength_lookback)),
            min_relative_strength=float(getattr(module, "ALPACA_MIN_RELATIVE_STRENGTH", cls.min_relative_strength)),
            min_entry_score=float(getattr(module, "ALPACA_MIN_ENTRY_SCORE", cls.min_entry_score)),
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
    market_regime_reason: str | None = None


class StrategyEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config

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

        if market_row["close"] <= market_row["ema_200"]:
            return False, f"{self.config.market_symbol} below EMA200"
        if benchmark_row["close"] <= benchmark_row["ema_200"]:
            return False, f"{self.config.benchmark_symbol} below EMA200"

        if volatility_daily is not None and not volatility_daily.empty:
            volatility_close = float(volatility_daily.iloc[-1]["close"])
            if volatility_close >= self.config.volatility_threshold:
                return False, f"volatility proxy above {self.config.volatility_threshold}"

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
            return SignalDecision("hold", f"{symbol}: incomplete OHLCV history", market_regime_reason=regime_reason)

        daily_row = daily_frame.iloc[-1]
        pullback_row = pullback_frame.iloc[-1]
        entry_row = entry_frame.iloc[-1]
        previous_entry_row = entry_frame.iloc[-2] if len(entry_frame) > 1 else None
        relative_strength = self.compute_relative_strength(daily_frame, benchmark_daily)

        if spread_pct is not None and spread_pct > self.config.max_spread_pct:
            return SignalDecision("hold", f"{symbol}: spread {spread_pct:.2f}% is too wide", spread_pct=spread_pct, relative_strength=relative_strength, market_regime_reason=regime_reason)

        if daily_row["avg_dollar_volume_20"] < self.config.min_avg_daily_dollar_volume:
            return SignalDecision("hold", f"{symbol}: dollar volume is too low", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if daily_row["close"] <= daily_row["ema_200"]:
            return SignalDecision("hold", f"{symbol}: close below EMA200", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if daily_row["ema_50"] <= daily_row["ema_200"]:
            return SignalDecision("hold", f"{symbol}: EMA50 below EMA200", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if daily_row["supertrend_direction"] != 1:
            return SignalDecision("hold", f"{symbol}: daily supertrend is not long", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if daily_row["adx"] < self.config.adx_threshold:
            return SignalDecision("hold", f"{symbol}: ADX below threshold", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if pd.isna(daily_row["atr"]) or daily_row["atr"] <= 0:
            return SignalDecision("hold", f"{symbol}: ATR unavailable", relative_strength=relative_strength, market_regime_reason=regime_reason)

        extension_atr = (daily_row["close"] - daily_row["ema_20"]) / daily_row["atr"]
        if extension_atr > self.config.max_extension_atr:
            return SignalDecision("hold", f"{symbol}: price is too extended from EMA20", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if daily_row["rsi"] > self.config.max_entry_rsi:
            return SignalDecision("hold", f"{symbol}: RSI is too high", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if relative_strength is not None and relative_strength < self.config.min_relative_strength:
            return SignalDecision("hold", f"{symbol}: relative strength is too weak", relative_strength=relative_strength, market_regime_reason=regime_reason)

        tolerance = self.config.pullback_tolerance_pct / 100
        touched_pullback_zone = (
            pullback_row["low"] <= pullback_row["ema_20"] * (1 + tolerance)
            or pullback_row["low"] <= pullback_row["ema_50"] * (1 + tolerance)
        )
        if not touched_pullback_zone:
            return SignalDecision("hold", f"{symbol}: no pullback to EMA20/EMA50 on {self.config.pullback_timeframe}", relative_strength=relative_strength, market_regime_reason=regime_reason)

        if previous_entry_row is None:
            return SignalDecision("hold", f"{symbol}: not enough entry bars", relative_strength=relative_strength, market_regime_reason=regime_reason)

        breakout = entry_row["close"] > previous_entry_row["high"]
        volume_ratio = float(entry_row["volume"] / entry_row["avg_volume_20"]) if entry_row["avg_volume_20"] and entry_row["avg_volume_20"] > 0 else 0.0
        volume_ok = volume_ratio >= self.config.volume_multiplier
        if not breakout:
            return SignalDecision("hold", f"{symbol}: no breakout on {self.config.entry_timeframe}", relative_strength=relative_strength, market_regime_reason=regime_reason)
        if not volume_ok:
            return SignalDecision("hold", f"{symbol}: breakout volume is too weak", relative_strength=relative_strength, market_regime_reason=regime_reason)

        signal_score = self.score_entry_setup(daily_row, pullback_row, entry_row, float(extension_atr), volume_ratio, relative_strength)
        if signal_score < self.config.min_entry_score:
            return SignalDecision("hold", f"{symbol}: setup score {signal_score:.2f} below threshold", signal_score=signal_score, relative_strength=relative_strength, market_regime_reason=regime_reason)

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
            return SignalDecision("hold", f"{symbol}: invalid stop distance", signal_score=signal_score, relative_strength=relative_strength, market_regime_reason=regime_reason)

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

        stored_stop = float(trade_state["stop_price"])
        atr_value = float(daily_row["atr"]) if not pd.isna(daily_row["atr"]) else trade_state.get("atr_value", 0.0)
        max_price = max(float(trade_state.get("max_price", current_price)), current_price)
        trailing_stop = max(
            stored_stop,
            max_price - atr_value * self.config.atr_trailing_multiplier if atr_value > 0 else stored_stop,
            float(daily_row["supertrend"]) if daily_row["supertrend_direction"] == 1 else stored_stop,
        )

        regime_ok, regime_reason = market_regime
        if current_price <= stored_stop:
            return SignalDecision("sell", f"{symbol}: hard stop hit", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if not regime_ok:
            return SignalDecision("sell", f"{symbol}: {regime_reason}", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if daily_row["supertrend_direction"] != 1:
            return SignalDecision("sell", f"{symbol}: daily supertrend flipped", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if daily_row["close"] < daily_row["ema_50"]:
            return SignalDecision("sell", f"{symbol}: daily close below EMA50", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if pullback_row["close"] < pullback_row["ema_50"]:
            return SignalDecision("sell", f"{symbol}: intraday close below EMA50", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)

        if not trade_state.get("partial_exit_done") and current_price >= float(trade_state["take_profit_price"]):
            return SignalDecision("partial_sell", f"{symbol}: first target reached", stop_price=stored_stop, trailing_stop_price=max(trailing_stop, float(trade_state["entry_price"])), atr_value=atr_value, market_regime_reason=regime_reason)

        return SignalDecision("hold", f"{symbol}: keep position", stop_price=stored_stop, trailing_stop_price=trailing_stop, atr_value=atr_value, market_regime_reason=regime_reason)
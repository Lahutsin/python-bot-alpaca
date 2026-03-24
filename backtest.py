from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import alpaca_trade_api as api
import numpy as np
import pandas as pd

import config
from indicators import bars_to_dataframe
from strategy import StrategyConfig, StrategyEngine


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    r_multiple: float
    bars_held: int
    signal_score: float
    relative_strength: float | None
    mfe_r: float
    mae_r: float


class TrendBacktester:
    def __init__(self) -> None:
        self.alpaca = api.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_URL, "v2")
        self.strategy_config = StrategyConfig.from_module(config)
        self.strategy = StrategyEngine(self.strategy_config)
        self.initial_equity = float(getattr(config, "ALPACA_BACKTEST_INITIAL_EQUITY", 100_000))
        self.daily_limit = int(getattr(config, "ALPACA_BACKTEST_DAILY_LIMIT", 900))
        self.pullback_limit = int(getattr(config, "ALPACA_BACKTEST_PULLBACK_LIMIT", 2500))
        self.entry_limit = int(getattr(config, "ALPACA_BACKTEST_ENTRY_LIMIT", 8000))
        self.frame_cache: dict[tuple[str, str, int], pd.DataFrame] = {}

    def fetch_frame(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        cache_key = (symbol, timeframe, limit)
        if cache_key not in self.frame_cache:
            bars = self.alpaca.get_bars(symbol, timeframe, limit=limit)
            self.frame_cache[cache_key] = self.strategy.prepare_frame(bars_to_dataframe(bars))
        return self.frame_cache[cache_key]

    def calculate_metrics(
        self,
        equity_curve: list[float],
        trades: list[TradeRecord],
        bars_with_position: int,
        total_bars: int,
    ) -> dict[str, float]:
        if not equity_curve:
            return {}

        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        wins = [trade.pnl for trade in trades if trade.pnl > 0]
        losses = [trade.pnl for trade in trades if trade.pnl < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        expectancy = np.mean([trade.r_multiple for trade in trades]) if trades else 0.0
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max

        return {
            "ending_equity": float(equity_series.iloc[-1]),
            "net_profit": float(equity_series.iloc[-1] - self.initial_equity),
            "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
            "average_win": float(np.mean(wins)) if wins else 0.0,
            "average_loss": float(np.mean(losses)) if losses else 0.0,
            "profit_factor": float(gross_profit / gross_loss) if gross_loss else 0.0,
            "expectancy_r": float(expectancy),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "sharpe": float((returns.mean() / returns.std()) * np.sqrt(252)) if not returns.empty and returns.std() else 0.0,
            "time_in_market": float(bars_with_position / total_bars) if total_bars else 0.0,
            "trades": float(len(trades)),
        }

    def update_performance_state(self, performance: dict[str, Any], r_multiple: float) -> None:
        if r_multiple < 0:
            performance["consecutive_losses"] += 1
        else:
            performance["consecutive_losses"] = 0

    def run_symbol_backtest(
        self,
        symbol: str,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        daily_frame = self.fetch_frame(symbol, self.strategy_config.daily_timeframe, self.daily_limit)
        pullback_frame = self.fetch_frame(symbol, self.strategy_config.pullback_timeframe, self.pullback_limit)
        entry_frame = self.fetch_frame(symbol, self.strategy_config.entry_timeframe, self.entry_limit)
        market_frame = self.fetch_frame(self.strategy_config.market_symbol, self.strategy_config.daily_timeframe, self.daily_limit)
        benchmark_frame = self.fetch_frame(self.strategy_config.benchmark_symbol, self.strategy_config.daily_timeframe, self.daily_limit)
        volatility_frame = None
        if self.strategy_config.volatility_symbol:
            volatility_frame = self.fetch_frame(self.strategy_config.volatility_symbol, self.strategy_config.daily_timeframe, self.daily_limit)

        if daily_frame.empty or pullback_frame.empty or entry_frame.empty:
            return {"symbol": symbol, "error": "not enough data"}

        active_start = pd.Timestamp(start_time) if start_time is not None else entry_frame.index[0]
        active_end = pd.Timestamp(end_time) if end_time is not None else entry_frame.index[-1]

        equity = self.initial_equity
        peak_equity = equity
        equity_curve: list[float] = []
        trades: list[TradeRecord] = []
        open_trade: dict[str, Any] | None = None
        bars_held = 0
        bars_with_position = 0
        performance = {"consecutive_losses": 0}

        for timestamp, entry_row in entry_frame.iterrows():
            if timestamp < active_start or timestamp > active_end:
                continue

            daily_slice = daily_frame.loc[:timestamp]
            pullback_slice = pullback_frame.loc[:timestamp]
            entry_slice = entry_frame.loc[:timestamp]
            market_slice = market_frame.loc[:timestamp]
            benchmark_slice = benchmark_frame.loc[:timestamp]
            volatility_slice = volatility_frame.loc[:timestamp] if volatility_frame is not None else None

            if len(daily_slice) < 220 or len(pullback_slice) < 50 or len(entry_slice) < 20:
                continue

            market_regime = self.strategy.evaluate_market_regime(market_slice, benchmark_slice, volatility_slice)
            price = float(entry_row["close"])

            if open_trade is None:
                decision = self.strategy.build_entry_decision(
                    symbol,
                    daily_slice,
                    pullback_slice,
                    entry_slice,
                    price,
                    spread_pct=0.0,
                    market_regime=market_regime,
                    benchmark_daily=benchmark_slice,
                )
                if decision.action == "buy" and decision.risk_per_share and decision.risk_per_share > 0:
                    effective_risk_pct = self.strategy.calculate_effective_risk_pct(equity, peak_equity, performance["consecutive_losses"])
                    risk_budget = equity * (effective_risk_pct / 100)
                    max_position_value = equity * (self.strategy_config.max_position_pct / 100)
                    quantity = int(min(risk_budget / decision.risk_per_share, max_position_value / price))
                    if quantity > 0:
                        open_trade = {
                            "entry_time": str(timestamp),
                            "entry_price": price,
                            "stop_price": float(decision.stop_price),
                            "risk_per_share": float(decision.risk_per_share),
                            "take_profit_price": float(decision.take_profit_price),
                            "quantity": quantity,
                            "remaining_quantity": quantity,
                            "partial_exit_done": False,
                            "realized_pnl": 0.0,
                            "max_price": price,
                            "min_price": price,
                            "signal_score": float(decision.signal_score),
                            "relative_strength": decision.relative_strength,
                        }
                        bars_held = 0
            else:
                bars_held += 1
                bars_with_position += 1
                open_trade["max_price"] = max(float(open_trade["max_price"]), price)
                open_trade["min_price"] = min(float(open_trade["min_price"]), price)
                exit_decision = self.strategy.build_exit_decision(symbol, open_trade, daily_slice, pullback_slice, price, market_regime)
                open_trade["stop_price"] = float(exit_decision.trailing_stop_price or open_trade["stop_price"])

                if exit_decision.action == "partial_sell" and not open_trade["partial_exit_done"]:
                    partial_quantity = max(1, int(open_trade["remaining_quantity"] * self.strategy_config.partial_exit_pct))
                    partial_pnl = (price - open_trade["entry_price"]) * partial_quantity
                    equity += partial_pnl
                    open_trade["realized_pnl"] += partial_pnl
                    open_trade["remaining_quantity"] -= partial_quantity
                    open_trade["partial_exit_done"] = True
                    open_trade["stop_price"] = max(float(open_trade["entry_price"]), float(open_trade["stop_price"]))

                elif exit_decision.action == "sell":
                    final_quantity = int(open_trade["remaining_quantity"])
                    final_pnl = (price - open_trade["entry_price"]) * final_quantity + float(open_trade["realized_pnl"])
                    equity += final_pnl
                    peak_equity = max(peak_equity, equity)
                    risk_amount = open_trade["risk_per_share"] * open_trade["quantity"]
                    mfe_r = (float(open_trade["max_price"]) - float(open_trade["entry_price"])) / float(open_trade["risk_per_share"])
                    mae_r = (float(open_trade["entry_price"]) - float(open_trade["min_price"])) / float(open_trade["risk_per_share"])
                    trade = TradeRecord(
                        entry_time=open_trade["entry_time"],
                        exit_time=str(timestamp),
                        entry_price=float(open_trade["entry_price"]),
                        exit_price=price,
                        quantity=int(open_trade["quantity"]),
                        pnl=final_pnl,
                        r_multiple=final_pnl / risk_amount if risk_amount else 0.0,
                        bars_held=bars_held,
                        signal_score=float(open_trade.get("signal_score", 0.0)),
                        relative_strength=open_trade.get("relative_strength"),
                        mfe_r=mfe_r,
                        mae_r=mae_r,
                    )
                    trades.append(trade)
                    self.update_performance_state(performance, trade.r_multiple)
                    open_trade = None

            equity_curve.append(equity)

        metrics = self.calculate_metrics(equity_curve, trades, bars_with_position, max(len(equity_curve), 1))
        return {
            "symbol": symbol,
            "metrics": metrics,
            "trades": [asdict(trade) for trade in trades],
        }

    def walk_forward_analysis(
        self,
        symbol: str,
        limit: int = 1250,
        train_bars: int = 252 * 2,
        test_bars: int = 126,
    ) -> list[dict[str, Any]]:
        daily_frame = self.fetch_frame(symbol, self.strategy_config.daily_timeframe, max(limit, self.daily_limit))
        if len(daily_frame) < 220 + train_bars + test_bars:
            return []

        windows = []
        start_index = 220
        while start_index + train_bars + test_bars <= len(daily_frame):
            train_start = daily_frame.index[start_index]
            train_end = daily_frame.index[start_index + train_bars - 1]
            test_start = daily_frame.index[start_index + train_bars]
            test_end = daily_frame.index[start_index + train_bars + test_bars - 1]

            train_report = self.run_symbol_backtest(symbol, train_start, train_end)
            test_report = self.run_symbol_backtest(symbol, test_start, test_end)
            windows.append(
                {
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "train": train_report.get("metrics", {}),
                    "test": test_report.get("metrics", {}),
                }
            )
            start_index += test_bars

        return windows


if __name__ == "__main__":
    symbols = list(config.ALPACA_STOCK_CONFIG.keys())
    runner = TrendBacktester()
    report = {
        "symbols": {symbol: runner.run_symbol_backtest(symbol) for symbol in symbols},
        "walk_forward": {symbol: runner.walk_forward_analysis(symbol) for symbol in symbols},
    }
    print(json.dumps(report, indent=2))
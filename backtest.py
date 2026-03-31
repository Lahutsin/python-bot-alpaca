from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import alpaca_trade_api as api
import numpy as np
import pandas as pd

import config
from indicators import bars_start_for_timeframe, bars_to_dataframe
from strategy import StrategyConfig, StrategyEngine


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_float(value: float, digits: int = 2) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def metric_bar(value: float, minimum: float, maximum: float, width: int = 24) -> str:
    if maximum <= minimum:
        return "-" * width
    clamped = max(minimum, min(maximum, value))
    ratio = (clamped - minimum) / (maximum - minimum)
    filled = int(round(ratio * width))
    return "#" * filled + "." * (width - filled)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "  ".join("-" * widths[index] for index in range(len(headers)))
    print(header_line)
    print(separator_line)
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def build_symbol_rows(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, symbol_payload in report_data["symbols"].items():
        metrics = symbol_payload.get("metrics", {})
        if not metrics:
            continue
        walk_forward_windows = report_data["walk_forward"].get(symbol, [])
        positive_tests = sum(1 for window in walk_forward_windows if float(window.get("test", {}).get("net_profit", 0.0)) > 0)
        rows.append(
            {
                "symbol": symbol,
                "trades": int(metrics.get("trades", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "net_profit": float(metrics.get("net_profit", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "expectancy_r": float(metrics.get("expectancy_r", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "time_in_market": float(metrics.get("time_in_market", 0.0)),
                "walk_forward_windows": len(walk_forward_windows),
                "walk_forward_positive_rate": (positive_tests / len(walk_forward_windows)) if walk_forward_windows else 0.0,
            }
        )
    return rows


def aggregate_summary(symbol_rows: list[dict[str, Any]], initial_equity: float) -> dict[str, float]:
    if not symbol_rows:
        return {}

    active_rows = [row for row in symbol_rows if row["trades"] > 0]
    rows_for_averages = active_rows or symbol_rows

    total_trades = sum(row["trades"] for row in symbol_rows)
    total_net_profit = sum(row["net_profit"] for row in symbol_rows)
    avg_win_rate = sum(row["win_rate"] for row in rows_for_averages) / len(rows_for_averages)
    avg_profit_factor = sum(row["profit_factor"] for row in rows_for_averages) / len(rows_for_averages)
    avg_expectancy = sum(row["expectancy_r"] for row in rows_for_averages) / len(rows_for_averages)
    worst_drawdown = min(row["max_drawdown"] for row in rows_for_averages)
    avg_sharpe = sum(row["sharpe"] for row in rows_for_averages) / len(rows_for_averages)
    avg_time_in_market = sum(row["time_in_market"] for row in rows_for_averages) / len(rows_for_averages)
    avg_walk_forward_positive_rate = sum(row["walk_forward_positive_rate"] for row in rows_for_averages) / len(rows_for_averages)

    return {
        "symbols": float(len(symbol_rows)),
        "active_symbols": float(len(active_rows)),
        "total_trades": float(total_trades),
        "total_net_profit": total_net_profit,
        "portfolio_return": (total_net_profit / initial_equity) if initial_equity else 0.0,
        "avg_win_rate": avg_win_rate,
        "avg_profit_factor": avg_profit_factor,
        "avg_expectancy_r": avg_expectancy,
        "worst_drawdown": worst_drawdown,
        "avg_sharpe": avg_sharpe,
        "avg_time_in_market": avg_time_in_market,
        "avg_walk_forward_positive_rate": avg_walk_forward_positive_rate,
    }


def walk_forward_summary(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, windows in report_data["walk_forward"].items():
        if not windows:
            continue
        test_metrics = [window.get("test", {}) for window in windows if window.get("test")]
        if not test_metrics:
            continue
        positive = [metrics for metrics in test_metrics if float(metrics.get("net_profit", 0.0)) > 0]
        rows.append(
            {
                "symbol": symbol,
                "windows": len(test_metrics),
                "positive_rate": len(positive) / len(test_metrics),
                "avg_test_profit": sum(float(metrics.get("net_profit", 0.0)) for metrics in test_metrics) / len(test_metrics),
                "avg_test_win_rate": sum(float(metrics.get("win_rate", 0.0)) for metrics in test_metrics) / len(test_metrics),
                "avg_test_pf": sum(float(metrics.get("profit_factor", 0.0)) for metrics in test_metrics) / len(test_metrics),
                "avg_test_drawdown": sum(float(metrics.get("max_drawdown", 0.0)) for metrics in test_metrics) / len(test_metrics),
            }
        )
    return rows


def format_date_label(value: str | None) -> str:
    if not value:
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def report_period(report_data: dict[str, Any]) -> tuple[str | None, str | None]:
    period = report_data.get("period", {})
    return period.get("start"), period.get("end")


def exit_reason_summary(report_data: dict[str, Any]) -> list[tuple[str, int]]:
    reason_counts: dict[str, int] = {}
    for reason_payload in report_data["symbols"].values():
        for reason_trade in reason_payload.get("trades", []):
            exit_reason = str(reason_trade.get("exit_reason", "unknown"))
            reason_counts[exit_reason] = reason_counts.get(exit_reason, 0) + 1
    return sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)


def readiness_notes(summary: dict[str, float]) -> list[str]:
    if not summary:
        return ["No completed backtest results."]

    notes: list[str] = []
    if summary["total_trades"] < 30:
        notes.append("Too few trades for confidence; increase sample size before trusting the edge.")
    if summary["avg_profit_factor"] < 1.15:
        notes.append("Profit factor is weak; strategy likely needs refinement before serious forward testing.")
    if summary["avg_expectancy_r"] <= 0:
        notes.append("Expectancy is non-positive; each trade is not earning enough for the risk taken.")
    if summary["worst_drawdown"] <= -0.15:
        notes.append("Drawdown is heavy; improve risk control before moving beyond paper testing.")
    if summary["avg_walk_forward_positive_rate"] < 0.55:
        notes.append("Walk-forward stability is weak; edge may not generalize out of sample.")
    if not notes:
        notes.append("Metrics are broadly healthy enough to justify paper or small-scale forward testing.")
    return notes


def print_terminal_report(report_data: dict[str, Any], initial_equity: float) -> None:
    symbol_rows = sorted(build_symbol_rows(report_data), key=lambda row: row["net_profit"], reverse=True)
    summary = aggregate_summary(symbol_rows, initial_equity)
    walk_rows = sorted(walk_forward_summary(report_data), key=lambda row: row["avg_test_profit"], reverse=True)
    exit_reasons = exit_reason_summary(report_data)
    period_start, period_end = report_period(report_data)

    if not symbol_rows:
        print("No completed backtest results.")
        return

    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Period: {format_date_label(period_start)} -> {format_date_label(period_end)}")
    print(f"Symbols: {int(summary['symbols'])} | Active: {int(summary['active_symbols'])} | Trades: {int(summary['total_trades'])} | Net PnL: {format_currency(summary['total_net_profit'])} | Return: {format_percent(summary['portfolio_return'])}")
    print(f"Avg win rate: {format_percent(summary['avg_win_rate'])} | Avg profit factor: {format_float(summary['avg_profit_factor'])} | Avg expectancy: {format_float(summary['avg_expectancy_r'])}R")
    print(f"Worst drawdown: {format_percent(summary['worst_drawdown'])} | Avg Sharpe: {format_float(summary['avg_sharpe'])} | Avg time in market: {format_percent(summary['avg_time_in_market'])}")
    print(f"Walk-forward positive windows: {format_percent(summary['avg_walk_forward_positive_rate'])}")
    print()

    print("SYMBOL SCORECARD")
    headers = ["Symbol", "Trades", "Win%", "Net PnL", "PF", "ExpR", "MaxDD", "WF+"]
    rows = [
        [
            row["symbol"],
            str(row["trades"]),
            format_percent(row["win_rate"]),
            format_currency(row["net_profit"]),
            format_float(row["profit_factor"]),
            format_float(row["expectancy_r"]),
            format_percent(row["max_drawdown"]),
            format_percent(row["walk_forward_positive_rate"]),
        ]
        for row in symbol_rows
    ]
    print_table(headers, rows)
    print()

    print("NET PNL BY SYMBOL")
    pnl_scale = max(max(abs(row["net_profit"]) for row in symbol_rows), 1.0)
    for row in symbol_rows:
        print(f"{row['symbol']:>5}  {metric_bar(row['net_profit'], -pnl_scale, pnl_scale)}  {format_currency(row['net_profit'])}")
    print()

    print("TRADES AND WIN RATE")
    max_trades = max(row["trades"] for row in symbol_rows) or 1
    for row in symbol_rows:
        trades_bar = metric_bar(float(row["trades"]), 0.0, float(max_trades))
        print(f"{row['symbol']:>5}  trades {trades_bar} {row['trades']:>3} | win {format_percent(row['win_rate'])}")
    print()

    if walk_rows:
        print("WALK-FORWARD TEST")
        walk_headers = ["Symbol", "Windows", "WF+", "Avg Test PnL", "Avg Test Win%", "Avg Test PF", "Avg Test DD"]
        walk_table_rows = [
            [
                row["symbol"],
                str(row["windows"]),
                format_percent(row["positive_rate"]),
                format_currency(row["avg_test_profit"]),
                format_percent(row["avg_test_win_rate"]),
                format_float(row["avg_test_pf"]),
                format_percent(row["avg_test_drawdown"]),
            ]
            for row in walk_rows
        ]
        print_table(walk_headers, walk_table_rows)
        print()

    if exit_reasons:
        print("EXIT REASONS")
        for reason, count in exit_reasons:
            print(f"{count:>4}  {reason}")
        print()

    print("READINESS")
    for note in readiness_notes(summary):
        print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the historical backtest and walk-forward report.")
    parser.add_argument("--json", action="store_true", help="Print the raw JSON report instead of the terminal summary.")
    return parser.parse_args()


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
    exit_reason: str


class TrendBacktester:
    def __init__(self) -> None:
        self.alpaca = api.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_URL, "v2")
        self.strategy_config = StrategyConfig.from_module(config)
        self.strategy = StrategyEngine(self.strategy_config)
        self.initial_equity = float(getattr(config, "ALPACA_BACKTEST_INITIAL_EQUITY", 100_000))
        self.backtest_years = float(getattr(config, "ALPACA_BACKTEST_YEARS", 2.0))
        self.daily_limit = int(getattr(config, "ALPACA_BACKTEST_DAILY_LIMIT", 900))
        self.pullback_limit = int(getattr(config, "ALPACA_BACKTEST_PULLBACK_LIMIT", 2500))
        self.entry_limit = int(getattr(config, "ALPACA_BACKTEST_ENTRY_LIMIT", 8000))
        self.frame_cache: dict[tuple[str, str, int], pd.DataFrame] = {}

    def default_backtest_period(self, entry_frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        active_end = pd.Timestamp(entry_frame.index[-1])
        if self.backtest_years > 0:
            active_start = active_end - pd.DateOffset(years=self.backtest_years)
        else:
            active_start = pd.Timestamp(entry_frame.index[0])
        active_start = max(active_start, pd.Timestamp(entry_frame.index[0]))
        return active_start, active_end

    def fetch_frame(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        cache_key = (symbol, timeframe, limit)
        if cache_key not in self.frame_cache:
            start = bars_start_for_timeframe(timeframe, limit)
            bars = self.alpaca.get_bars(symbol, timeframe, start=start, limit=limit)
            self.frame_cache[cache_key] = self.strategy.prepare_frame(bars_to_dataframe(bars))
        return self.frame_cache[cache_key]

    def symbol_allocation_pct(self, symbol: str) -> float:
        configured_allocations = getattr(config, "ALPACA_STOCK_CONFIG", {})
        configured_pct = configured_allocations.get(symbol, self.strategy_config.max_position_pct)
        return max(0.0, float(configured_pct))

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
        profit_factor = float(gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit > 0 else 0.0)

        return {
            "ending_equity": float(equity_series.iloc[-1]),
            "net_profit": float(equity_series.iloc[-1] - self.initial_equity),
            "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
            "average_win": float(np.mean(wins)) if wins else 0.0,
            "average_loss": float(np.mean(losses)) if losses else 0.0,
            "profit_factor": profit_factor,
            "expectancy_r": float(expectancy),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "sharpe": float((returns.mean() / returns.std()) * np.sqrt(252)) if not returns.empty and returns.std() else 0.0,
            "time_in_market": float(bars_with_position / total_bars) if total_bars else 0.0,
            "trades": float(len(trades)),
        }

    def finalize_trade(
        self,
        open_trade: dict[str, Any],
        timestamp: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord:
        final_quantity = int(open_trade["remaining_quantity"])
        final_pnl = (exit_price - open_trade["entry_price"]) * final_quantity + float(open_trade["realized_pnl"])
        risk_amount = open_trade["risk_per_share"] * open_trade["quantity"]
        mfe_r = (float(open_trade["max_price"]) - float(open_trade["entry_price"])) / float(open_trade["risk_per_share"])
        mae_r = (float(open_trade["entry_price"]) - float(open_trade["min_price"])) / float(open_trade["risk_per_share"])
        return TradeRecord(
            entry_time=open_trade["entry_time"],
            exit_time=str(timestamp),
            entry_price=float(open_trade["entry_price"]),
            exit_price=exit_price,
            quantity=int(open_trade["quantity"]),
            pnl=final_pnl,
            r_multiple=final_pnl / risk_amount if risk_amount else 0.0,
            bars_held=int(open_trade.get("bars_held", 0)),
            signal_score=float(open_trade.get("signal_score", 0.0)),
            relative_strength=open_trade.get("relative_strength"),
            mfe_r=mfe_r,
            mae_r=mae_r,
            exit_reason=exit_reason,
        )

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

        default_start, default_end = self.default_backtest_period(entry_frame)
        active_start = pd.Timestamp(start_time) if start_time is not None else default_start
        active_end = pd.Timestamp(end_time) if end_time is not None else default_end

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
            bar_high = float(entry_row["high"])
            bar_low = float(entry_row["low"])

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
                    allocation_pct = self.symbol_allocation_pct(symbol)
                    target_position_value = equity * (allocation_pct / 100)
                    quantity = int(target_position_value / price)
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
                assert open_trade is not None
                current_trade = open_trade
                bars_held += 1
                bars_with_position += 1
                current_trade["bars_held"] = bars_held
                current_trade["max_price"] = max(float(current_trade["max_price"]), bar_high)
                current_trade["min_price"] = min(float(current_trade["min_price"]), bar_low)
                exit_decision = self.strategy.build_exit_decision(symbol, current_trade, daily_slice, pullback_slice, price, market_regime)
                current_trade["stop_price"] = float(exit_decision.trailing_stop_price or current_trade["stop_price"])
                trailing_stop_price = float(exit_decision.trailing_stop_price or current_trade["stop_price"])

                if bar_low <= trailing_stop_price:
                    realized_trade = self.finalize_trade(
                        current_trade,
                        timestamp,
                        trailing_stop_price,
                        f"{symbol}: {'trailing stop hit' if trailing_stop_price > float(current_trade['entry_price']) - float(current_trade['risk_per_share']) else 'hard stop hit'}",
                    )
                    equity += realized_trade.pnl
                    peak_equity = max(peak_equity, equity)
                    trades.append(realized_trade)
                    self.update_performance_state(performance, realized_trade.r_multiple)
                    open_trade = None
                    equity_curve.append(equity)
                    continue

                if not current_trade["partial_exit_done"] and bar_high >= float(current_trade["take_profit_price"]):
                    partial_quantity = max(1, int(current_trade["remaining_quantity"] * self.strategy_config.partial_exit_pct))
                    partial_fill_price = float(current_trade["take_profit_price"])
                    partial_pnl = (partial_fill_price - current_trade["entry_price"]) * partial_quantity
                    equity += partial_pnl
                    current_trade["realized_pnl"] += partial_pnl
                    current_trade["remaining_quantity"] -= partial_quantity
                    current_trade["partial_exit_done"] = True
                    current_trade["stop_price"] = max(float(current_trade["entry_price"]), float(current_trade["stop_price"]))

                elif exit_decision.action == "sell":
                    realized_trade = self.finalize_trade(current_trade, timestamp, price, exit_decision.reason)
                    equity += realized_trade.pnl
                    peak_equity = max(peak_equity, equity)
                    trades.append(realized_trade)
                    self.update_performance_state(performance, realized_trade.r_multiple)
                    open_trade = None

            equity_curve.append(equity)

        metrics = self.calculate_metrics(equity_curve, trades, bars_with_position, max(len(equity_curve), 1))
        return {
            "symbol": symbol,
            "metrics": metrics,
            "trades": [asdict(trade) for trade in trades],
            "period": {
                "start": str(active_start),
                "end": str(active_end),
            },
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
    args = parse_args()
    symbols = list(config.ALPACA_STOCK_CONFIG.keys())
    runner = TrendBacktester()
    report = {
        "symbols": {symbol: runner.run_symbol_backtest(symbol) for symbol in symbols},
        "walk_forward": {symbol: runner.walk_forward_analysis(symbol) for symbol in symbols},
        "period": {
            "start": None,
            "end": None,
        },
    }
    period_starts = []
    period_ends = []
    for report_payload in report["symbols"].values():
        symbol_period = report_payload.get("period", {})
        if symbol_period.get("start"):
            period_starts.append(str(symbol_period["start"]))
        if symbol_period.get("end"):
            period_ends.append(str(symbol_period["end"]))
    if period_starts and period_ends:
        report["period"]["start"] = min(period_starts)
        report["period"]["end"] = max(period_ends)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_terminal_report(report, runner.initial_equity)
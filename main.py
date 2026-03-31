from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import alpaca_trade_api as api
import pandas as pd

import config
from indicators import bars_start_for_timeframe, bars_to_dataframe
from strategy import StrategyConfig, StrategyEngine


class StockBot:
    JOURNAL_HEADER = [
        "timestamp",
        "event",
        "symbol",
        "side",
        "quantity",
        "price",
        "reason",
        "atr",
        "stop_price",
        "risk_amount",
        "r_multiple",
        "signal_score",
        "relative_strength",
        "effective_risk_pct",
        "market_regime",
        "mfe_r",
        "mae_r",
        "notes",
    ]

    @staticmethod
    def resolve_runtime_path(env_name: str, config_name: str, default_value: str) -> Path:
        configured_value = os.getenv(env_name) or getattr(config, config_name, default_value)
        path = Path(configured_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def __init__(self) -> None:
        self.alpaca = api.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_URL, "v2")
        self.symbols = config.ALPACA_STOCK_CONFIG
        self.bot_version = config.ALPACA_STOCKING_BOT_VERSION
        self.sleeper = int(config.ALPACA_SLEEP_TIMEOUT)
        self.strategy_config = StrategyConfig.from_module(config)
        self.strategy = StrategyEngine(self.strategy_config)
        self.state_path = self.resolve_runtime_path("ALPACA_STATE_PATH", "ALPACA_STATE_PATH", "bot_state.json")
        self.journal_path = self.resolve_runtime_path("ALPACA_TRADE_JOURNAL", "ALPACA_TRADE_JOURNAL", "trade_journal.csv")
        self.log_path = self.resolve_runtime_path("ALPACA_LOG_PATH", "ALPACA_LOG_PATH", "debug.log")
        self.heartbeat_path = self.resolve_runtime_path("ALPACA_HEARTBEAT_PATH", "ALPACA_HEARTBEAT_PATH", "bot_heartbeat.json")

        log_format = "%(levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(filename=self.log_path, filemode="a", format=log_format, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.state = self.load_state()
        self.ensure_journal_exists()
        self.write_heartbeat("starting")

    def write_heartbeat(self, status: str, error: str = "") -> None:
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "pid": os.getpid(),
            "status": status,
            "bot_version": self.bot_version,
            "sleep_timeout_minutes": self.sleeper,
            "error": error,
        }
        temp_path = self.heartbeat_path.with_suffix(self.heartbeat_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(payload, file_pointer, indent=2, sort_keys=True)
        temp_path.replace(self.heartbeat_path)

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as file_pointer:
                return json.load(file_pointer)
        return {"open_trades": {}, "risk_windows": {}, "performance": {"consecutive_losses": 0, "paused_until": None}}

    def save_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(self.state, file_pointer, indent=2, sort_keys=True)

    def performance_state(self) -> dict[str, Any]:
        return self.state.setdefault("performance", {"consecutive_losses": 0, "paused_until": None})

    def ensure_journal_exists(self) -> None:
        header_prefix = ",".join(self.JOURNAL_HEADER)
        if not self.journal_path.exists():
            with self.journal_path.open("w", encoding="utf-8", newline="") as file_pointer:
                writer = csv.writer(file_pointer)
                writer.writerow(self.JOURNAL_HEADER)
            return

        with self.journal_path.open("r", encoding="utf-8", newline="") as file_pointer:
            existing_content = file_pointer.read()

        if not existing_content:
            with self.journal_path.open("w", encoding="utf-8", newline="") as file_pointer:
                writer = csv.writer(file_pointer)
                writer.writerow(self.JOURNAL_HEADER)
            return

        first_line = existing_content.splitlines()[0].strip()
        if first_line != header_prefix:
            with self.journal_path.open("w", encoding="utf-8", newline="") as file_pointer:
                file_pointer.write(header_prefix + "\n")
                file_pointer.write(existing_content)

    def append_journal(
        self,
        event: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        reason: str,
        atr_value: float,
        stop_price: float,
        risk_amount: float,
        r_multiple: float = 0.0,
        signal_score: float = 0.0,
        relative_strength: float | None = None,
        effective_risk_pct: float = 0.0,
        market_regime: str = "",
        mfe_r: float = 0.0,
        mae_r: float = 0.0,
        notes: str = "",
    ) -> None:
        with self.journal_path.open("a", encoding="utf-8", newline="") as file_pointer:
            writer = csv.writer(file_pointer)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    event,
                    symbol,
                    side,
                    quantity,
                    round(price, 4),
                    reason,
                    round(atr_value, 4),
                    round(stop_price, 4),
                    round(risk_amount, 4),
                    round(r_multiple, 4),
                    round(signal_score, 4),
                    "" if relative_strength is None else round(relative_strength, 4),
                    round(effective_risk_pct, 4),
                    market_regime,
                    round(mfe_r, 4),
                    round(mae_r, 4),
                    notes,
                ]
            )

    def append_entry_skip(
        self,
        symbol: str,
        reason: str,
        *,
        price: float = 0.0,
        signal_score: float = 0.0,
        relative_strength: float | None = None,
        market_regime: str = "",
        notes: str = "",
    ) -> None:
        self.append_journal(
            "entry_skip",
            symbol,
            "hold",
            0,
            price,
            reason,
            0.0,
            0.0,
            0.0,
            signal_score=signal_score,
            relative_strength=relative_strength,
            market_regime=market_regime,
            notes=notes,
        )

    def refresh_account(self):
        return self.alpaca.get_account()

    def wait_for_market_open(self) -> None:
        clock = self.alpaca.get_clock()
        while not clock.is_open:
            next_open_time = clock.next_open.timestamp()
            current_time = clock.timestamp.timestamp()
            difference_minutes = max((int(next_open_time) - int(current_time)) / 60, 1)
            hours = round(difference_minutes // 60)
            remaining_minutes = round(difference_minutes % 60)
            self.logger.info("Market closed. Sleeping for %s hours and %s minutes.", hours, remaining_minutes)
            time.sleep((difference_minutes + 5) * 60)
            clock = self.alpaca.get_clock()

    def fetch_frame(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        start = bars_start_for_timeframe(timeframe, limit)
        bars = self.alpaca.get_bars(symbol, timeframe, start=start, limit=limit)
        return self.strategy.prepare_frame(bars_to_dataframe(bars))

    def get_mid_price(self, symbol: str, fallback_price: float | None = None) -> tuple[float, float | None]:
        quote = self.alpaca.get_latest_quote(symbol)
        ask_price = float(getattr(quote, "ap", 0.0) or getattr(quote, "ask_price", 0.0) or 0.0)
        bid_price = float(getattr(quote, "bp", 0.0) or getattr(quote, "bid_price", 0.0) or 0.0)
        if ask_price > 0 and bid_price > 0:
            mid_price = (ask_price + bid_price) / 2
            spread_pct = ((ask_price - bid_price) / mid_price) * 100 if mid_price else None
            return mid_price, spread_pct
        if ask_price > 0:
            return ask_price, None
        if bid_price > 0:
            return bid_price, None
        if fallback_price is not None:
            return fallback_price, None
        raise ValueError(f"Unable to determine price for {symbol}")

    def get_market_context(self) -> dict[str, Any]:
        market_frame = self.fetch_frame(self.strategy_config.market_symbol, self.strategy_config.daily_timeframe, 260)
        benchmark_frame = self.fetch_frame(self.strategy_config.benchmark_symbol, self.strategy_config.daily_timeframe, 260)
        volatility_frame = None
        if self.strategy_config.volatility_symbol:
            volatility_frame = self.fetch_frame(self.strategy_config.volatility_symbol, self.strategy_config.daily_timeframe, 260)
        regime = self.strategy.evaluate_market_regime(market_frame, benchmark_frame, volatility_frame)
        return {
            "regime": regime,
            "market_frame": market_frame,
            "benchmark_frame": benchmark_frame,
            "volatility_frame": volatility_frame,
        }

    def refresh_risk_windows(self, equity: float) -> None:
        today = datetime.utcnow().date().isoformat()
        iso_year, iso_week, _ = datetime.utcnow().isocalendar()
        current_week = f"{iso_year}-W{iso_week}"
        risk_windows = self.state.setdefault("risk_windows", {})
        daily_window = risk_windows.get("daily")
        weekly_window = risk_windows.get("weekly")

        if not daily_window or daily_window.get("date") != today:
            risk_windows["daily"] = {"date": today, "start_equity": equity}
        if not weekly_window or weekly_window.get("week") != current_week:
            risk_windows["weekly"] = {"week": current_week, "start_equity": equity}

    def risk_limits_breached(self, equity: float) -> tuple[bool, str]:
        self.refresh_risk_windows(equity)
        daily_start = float(self.state["risk_windows"]["daily"]["start_equity"])
        weekly_start = float(self.state["risk_windows"]["weekly"]["start_equity"])
        daily_drawdown = ((equity - daily_start) / daily_start) * 100 if daily_start else 0.0
        weekly_drawdown = ((equity - weekly_start) / weekly_start) * 100 if weekly_start else 0.0

        if daily_drawdown <= -self.strategy_config.daily_loss_limit_pct:
            return True, f"daily loss limit reached: {daily_drawdown:.2f}%"
        if weekly_drawdown <= -self.strategy_config.weekly_loss_limit_pct:
            return True, f"weekly loss limit reached: {weekly_drawdown:.2f}%"
        return False, "risk limits are fine"

    def entries_paused(self) -> tuple[bool, str]:
        paused_until = self.performance_state().get("paused_until")
        if not paused_until:
            return False, ""
        paused_until_ts = datetime.fromisoformat(paused_until)
        if datetime.utcnow() < paused_until_ts:
            return True, f"cooldown active until {paused_until_ts.isoformat()}"
        self.performance_state()["paused_until"] = None
        return False, ""

    def update_performance_after_exit(self, r_multiple: float) -> None:
        performance = self.performance_state()
        if r_multiple < 0:
            performance["consecutive_losses"] = int(performance.get("consecutive_losses", 0)) + 1
        else:
            performance["consecutive_losses"] = 0

        if performance["consecutive_losses"] >= self.strategy_config.max_consecutive_losses:
            performance["paused_until"] = (datetime.utcnow() + timedelta(minutes=self.strategy_config.cooldown_minutes)).isoformat()

    def effective_risk_pct(self, equity: float) -> float:
        self.refresh_risk_windows(equity)
        weekly_start = float(self.state["risk_windows"]["weekly"]["start_equity"])
        consecutive_losses = int(self.performance_state().get("consecutive_losses", 0))
        return self.strategy.calculate_effective_risk_pct(equity, weekly_start, consecutive_losses)

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> tuple[int, float, float]:
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0, 0.0, 0.0

        effective_risk_pct = self.effective_risk_pct(equity)
        risk_budget = equity * (effective_risk_pct / 100)
        max_position_value = equity * (self.strategy_config.max_position_pct / 100)
        quantity_by_risk = math.floor(risk_budget / risk_per_share)
        quantity_by_value = math.floor(max_position_value / entry_price)
        quantity = max(0, min(quantity_by_risk, quantity_by_value))
        return quantity, risk_budget, effective_risk_pct

    def get_positions(self) -> dict[str, Any]:
        return {position.symbol: position for position in self.alpaca.list_positions()}

    def remove_closed_positions_from_state(self, live_positions: dict[str, Any]) -> None:
        open_trades = self.state.setdefault("open_trades", {})
        stale_symbols = [symbol for symbol in open_trades if symbol not in live_positions]
        for symbol in stale_symbols:
            del open_trades[symbol]

    def correlated_position_count(self, candidate_symbol: str, open_symbols: list[str]) -> int:
        if not open_symbols:
            return 0

        candidate_frame = self.fetch_frame(candidate_symbol, self.strategy_config.daily_timeframe, self.strategy_config.correlation_lookback + 20)
        if candidate_frame.empty:
            return 0
        candidate_returns = candidate_frame["close"].pct_change().dropna()

        correlated = 0
        for symbol in open_symbols:
            comparison_frame = self.fetch_frame(symbol, self.strategy_config.daily_timeframe, self.strategy_config.correlation_lookback + 20)
            if comparison_frame.empty:
                continue
            comparison_returns = comparison_frame["close"].pct_change().dropna()
            combined = pd.concat([candidate_returns, comparison_returns], axis=1, join="inner").dropna().tail(self.strategy_config.correlation_lookback)
            if len(combined) < 20:
                continue
            correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if correlation is not None and correlation >= self.strategy_config.correlation_threshold:
                correlated += 1
        return correlated

    def submit_market_order(self, symbol: str, quantity: int, side: str) -> None:
        self.alpaca.submit_order(symbol=symbol, qty=quantity, side=side, type="market", time_in_force="day")

    def initialise_trade_state(self, position: Any, daily_frame: pd.DataFrame, current_price: float) -> dict[str, Any]:
        daily_row = daily_frame.iloc[-1]
        atr_value = float(daily_row["atr"]) if not pd.isna(daily_row["atr"]) else current_price * 0.02
        entry_price = float(position.avg_entry_price)
        stop_price = entry_price - atr_value * self.strategy_config.atr_stop_multiplier
        risk_per_share = max(entry_price - stop_price, 0.01)
        quantity = int(float(position.qty))
        return {
            "entry_price": entry_price,
            "stop_price": stop_price,
            "take_profit_price": entry_price + risk_per_share * self.strategy_config.partial_take_profit_r,
            "quantity": quantity,
            "remaining_quantity": quantity,
            "partial_exit_done": False,
            "max_price": max(entry_price, current_price),
            "min_price": min(entry_price, current_price),
            "risk_per_share": risk_per_share,
            "atr_value": atr_value,
            "signal_score": 0.0,
            "relative_strength": None,
        }

    def append_synced_entry(self, symbol: str, trade_state: dict[str, Any], market_regime_reason: str) -> None:
        quantity = int(float(trade_state.get("quantity", 0)))
        risk_per_share = float(trade_state.get("risk_per_share", 0.0))
        self.append_journal(
            "entry_sync",
            symbol,
            "buy",
            quantity,
            float(trade_state.get("entry_price", 0.0)),
            f"{symbol}: synchronized existing broker position",
            float(trade_state.get("atr_value", 0.0)),
            float(trade_state.get("stop_price", 0.0)),
            risk_per_share * quantity,
            signal_score=float(trade_state.get("signal_score", 0.0)),
            relative_strength=trade_state.get("relative_strength"),
            market_regime=market_regime_reason,
            notes="synced from live position",
        )

    def mfe_mae_r(self, trade_state: dict[str, Any]) -> tuple[float, float]:
        risk_per_share = float(trade_state.get("risk_per_share", 0.0))
        if risk_per_share <= 0:
            return 0.0, 0.0
        entry_price = float(trade_state["entry_price"])
        max_price = float(trade_state.get("max_price", entry_price))
        min_price = float(trade_state.get("min_price", entry_price))
        mfe_r = (max_price - entry_price) / risk_per_share
        mae_r = (entry_price - min_price) / risk_per_share
        return mfe_r, mae_r

    def manage_open_positions(self, live_positions: dict[str, Any], market_context: dict[str, Any]) -> None:
        open_trades = self.state.setdefault("open_trades", {})
        regime = market_context["regime"]

        for symbol, position in live_positions.items():
            daily_frame = self.fetch_frame(symbol, self.strategy_config.daily_timeframe, 260)
            pullback_frame = self.fetch_frame(symbol, self.strategy_config.pullback_timeframe, 200)
            fallback_price = float(daily_frame.iloc[-1]["close"]) if not daily_frame.empty else float(position.current_price)
            current_price, _ = self.get_mid_price(symbol, fallback_price)

            trade_state = open_trades.get(symbol)
            if trade_state is None:
                trade_state = self.initialise_trade_state(position, daily_frame, current_price)
                open_trades[symbol] = trade_state
                self.append_synced_entry(symbol, trade_state, regime[1])

            trade_state["max_price"] = max(float(trade_state.get("max_price", current_price)), current_price)
            trade_state["min_price"] = min(float(trade_state.get("min_price", current_price)), current_price)
            decision = self.strategy.build_exit_decision(symbol, trade_state, daily_frame, pullback_frame, current_price, regime)
            trade_state["stop_price"] = float(decision.trailing_stop_price or trade_state["stop_price"])
            trade_state["atr_value"] = float(decision.atr_value or trade_state.get("atr_value", 0.0))
            mfe_r, mae_r = self.mfe_mae_r(trade_state)

            if decision.action == "partial_sell" and not trade_state.get("partial_exit_done"):
                partial_quantity = max(1, int(float(position.qty) * self.strategy_config.partial_exit_pct))
                partial_quantity = min(partial_quantity, int(float(position.qty)))
                if partial_quantity > 0:
                    self.submit_market_order(symbol, partial_quantity, "sell")
                    trade_state["partial_exit_done"] = True
                    trade_state["remaining_quantity"] = max(0, int(float(position.qty)) - partial_quantity)
                    trade_state["stop_price"] = max(float(trade_state["entry_price"]), float(trade_state["stop_price"]))
                    risk_amount = float(trade_state["risk_per_share"]) * partial_quantity
                    r_multiple = ((current_price - float(trade_state["entry_price"])) * partial_quantity) / risk_amount if risk_amount else 0.0
                    self.append_journal(
                        "partial_exit",
                        symbol,
                        "sell",
                        partial_quantity,
                        current_price,
                        decision.reason,
                        float(trade_state["atr_value"]),
                        float(trade_state["stop_price"]),
                        risk_amount,
                        r_multiple,
                        signal_score=float(trade_state.get("signal_score", 0.0)),
                        relative_strength=trade_state.get("relative_strength"),
                        market_regime=regime[1],
                        mfe_r=mfe_r,
                        mae_r=mae_r,
                        notes="partial profit",
                    )

            elif decision.action == "sell":
                exit_quantity = int(float(position.qty))
                if exit_quantity > 0:
                    self.submit_market_order(symbol, exit_quantity, "sell")
                    risk_amount = float(trade_state["risk_per_share"]) * int(float(trade_state.get("quantity", exit_quantity)))
                    realized = (current_price - float(trade_state["entry_price"])) * exit_quantity
                    r_multiple = realized / risk_amount if risk_amount else 0.0
                    self.append_journal(
                        "exit",
                        symbol,
                        "sell",
                        exit_quantity,
                        current_price,
                        decision.reason,
                        float(trade_state["atr_value"]),
                        float(trade_state["stop_price"]),
                        risk_amount,
                        r_multiple,
                        signal_score=float(trade_state.get("signal_score", 0.0)),
                        relative_strength=trade_state.get("relative_strength"),
                        market_regime=regime[1],
                        mfe_r=mfe_r,
                        mae_r=mae_r,
                        notes="full exit",
                    )
                    self.update_performance_after_exit(r_multiple)
                    del open_trades[symbol]

    def evaluate_new_entries(self, account: Any, live_positions: dict[str, Any], market_context: dict[str, Any]) -> None:
        open_trades = self.state.setdefault("open_trades", {})
        regime = market_context["regime"]
        benchmark_frame = market_context["benchmark_frame"]
        equity = float(account.equity)
        open_symbols = list(live_positions.keys())
        regime_ok, regime_reason = regime

        if not regime_ok:
            self.logger.info("Skipping new entries: %s", regime_reason)
            self.append_entry_skip("PORTFOLIO", regime_reason, market_regime=regime_reason, notes="market regime gate")
            return

        if len(open_symbols) >= self.strategy_config.max_open_positions:
            self.logger.info("Skipping new entries: max open positions reached.")
            self.append_entry_skip("PORTFOLIO", "max open positions reached", market_regime=regime[1], notes="portfolio gate")
            return

        limits_hit, limits_reason = self.risk_limits_breached(equity)
        if limits_hit:
            self.logger.warning("Skipping new entries: %s", limits_reason)
            self.append_entry_skip("PORTFOLIO", limits_reason, market_regime=regime[1], notes="risk gate")
            return

        paused, pause_reason = self.entries_paused()
        if paused:
            self.logger.warning("Skipping new entries: %s", pause_reason)
            self.append_entry_skip("PORTFOLIO", pause_reason, market_regime=regime[1], notes="cooldown gate")
            return

        candidates: list[dict[str, Any]] = []

        for symbol in self.symbols:
            if symbol in live_positions or symbol in open_trades:
                self.append_entry_skip(symbol, "position already open", market_regime=regime[1], notes="already tracked")
                continue

            correlated_count = self.correlated_position_count(symbol, open_symbols)
            if correlated_count >= self.strategy_config.max_correlated_positions:
                self.logger.info("Skipping %s: correlation limit reached.", symbol)
                self.append_entry_skip(symbol, "correlation limit reached", market_regime=regime[1], notes=f"correlated positions: {correlated_count}")
                continue

            daily_frame = self.fetch_frame(symbol, self.strategy_config.daily_timeframe, 260)
            pullback_frame = self.fetch_frame(symbol, self.strategy_config.pullback_timeframe, 200)
            entry_frame = self.fetch_frame(symbol, self.strategy_config.entry_timeframe, 200)
            fallback_price = float(entry_frame.iloc[-1]["close"]) if not entry_frame.empty else None
            current_price, spread_pct = self.get_mid_price(symbol, fallback_price)

            decision = self.strategy.build_entry_decision(
                symbol,
                daily_frame,
                pullback_frame,
                entry_frame,
                current_price,
                spread_pct,
                regime,
                benchmark_daily=benchmark_frame,
            )
            if decision.action != "buy":
                self.logger.info(decision.reason)
                self.append_entry_skip(
                    symbol,
                    decision.reason,
                    price=current_price,
                    signal_score=float(decision.signal_score),
                    relative_strength=decision.relative_strength,
                    market_regime=decision.market_regime_reason or regime[1],
                    notes="strategy rejected entry",
                )
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "decision": decision,
                }
            )

        candidates.sort(
            key=lambda candidate: (
                float(candidate["decision"].signal_score),
                float(candidate["decision"].relative_strength or -999),
            ),
            reverse=True,
        )

        for candidate in candidates:
            if len(open_symbols) >= self.strategy_config.max_open_positions:
                break

            symbol = candidate["symbol"]
            decision = candidate["decision"]
            quantity, risk_budget, effective_risk_pct = self.position_size(equity, float(decision.entry_price), float(decision.stop_price))
            if quantity < 1:
                self.logger.info("Skipping %s: quantity calculated as zero.", symbol)
                self.append_entry_skip(
                    symbol,
                    "quantity calculated as zero",
                    price=float(decision.entry_price),
                    signal_score=float(decision.signal_score),
                    relative_strength=decision.relative_strength,
                    market_regime=regime[1],
                    notes="risk budget too small for stop distance",
                )
                continue

            self.submit_market_order(symbol, quantity, "buy")
            open_symbols.append(symbol)
            open_trades[symbol] = {
                "entry_price": float(decision.entry_price),
                "stop_price": float(decision.stop_price),
                "take_profit_price": float(decision.take_profit_price),
                "quantity": quantity,
                "remaining_quantity": quantity,
                "partial_exit_done": False,
                "max_price": float(decision.entry_price),
                "min_price": float(decision.entry_price),
                "risk_per_share": float(decision.risk_per_share),
                "atr_value": float(decision.atr_value or 0.0),
                "signal_score": float(decision.signal_score),
                "relative_strength": decision.relative_strength,
            }
            self.append_journal(
                "entry",
                symbol,
                "buy",
                quantity,
                float(decision.entry_price),
                decision.reason,
                float(decision.atr_value or 0.0),
                float(decision.stop_price),
                risk_budget,
                signal_score=float(decision.signal_score),
                relative_strength=decision.relative_strength,
                effective_risk_pct=effective_risk_pct,
                market_regime=regime[1],
                notes="ranked entry",
            )

    def run_cycle(self) -> None:
        self.write_heartbeat("running")
        self.logger.info("Stocking Bot v%s running.", self.bot_version)
        self.wait_for_market_open()
        account = self.refresh_account()

        if account.trading_blocked:
            self.logger.warning("Account is currently restricted from trading.")
            return

        market_context = self.get_market_context()
        live_positions = self.get_positions()
        self.remove_closed_positions_from_state(live_positions)
        self.manage_open_positions(live_positions, market_context)
        time.sleep(1)
        refreshed_account = self.refresh_account()
        refreshed_positions = self.get_positions()
        self.evaluate_new_entries(refreshed_account, refreshed_positions, market_context)
        self.save_state()
        self.write_heartbeat("sleeping")

    def bot_loop(self) -> None:
        while True:
            try:
                self.run_cycle()
            except (api.rest.APIError, ValueError, KeyError, TypeError) as exc:
                self.write_heartbeat("error", str(exc))
                self.logger.exception("Bot cycle failed: %s", exc)
            self.write_heartbeat("sleeping")
            time.sleep(self.sleeper * 60)


if __name__ == "__main__":
    StockBot().bot_loop()
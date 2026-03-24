from __future__ import annotations

import json

from alpaca_trade_api import REST

import config
from indicators import bars_to_dataframe, supertrend


class SuperTrend:
    def __init__(self) -> None:
        self.alpaca = REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_URL, "v2")
        self.symbols = config.ALPACA_STOCK_CONFIG
        self.data_timeframe = getattr(config, "ALPACA_DAILY_TIMEFRAME", "1Day")
        self.length = int(getattr(config, "ALPACA_SUPERTREND_LENGTH", 10))
        self.multiplier = float(getattr(config, "ALPACA_SUPERTREND_MULTIPLIER", 3.0))

    def get_supertrend_signal(self, symbol: str) -> dict[str, float | int | str]:
        bars = self.alpaca.get_bars(symbol, self.data_timeframe, limit=max(self.length * 5, 100))
        frame = bars_to_dataframe(bars)
        if frame.empty:
            return {"symbol": symbol, "signal": "no_data"}

        supertrend_frame = supertrend(frame, self.length, self.multiplier)
        latest = supertrend_frame.iloc[-1]
        return {
            "symbol": symbol,
            "signal": "buy" if int(latest["supertrend_direction"]) == 1 else "sell",
            "supertrend": float(latest["supertrend"]),
            "direction": int(latest["supertrend_direction"]),
        }

    def run(self) -> None:
        payload = {symbol: self.get_supertrend_signal(symbol) for symbol in self.symbols}
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    SuperTrend().run()
import logging
from alpaca_trade_api import REST

import config

class SuperTrend:
    def __init__(self):
        self.alpaca = REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_URL, "v2")
        self.symbols = config.ALPACA_STOCK_CONFIG
        self.data_timeframe = "1Day"

        log_format = "%(levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(filename = "debug_supertrend.log", filemode = "w", format = log_format, level = logging.DEBUG)

    def getSuperTrend(self, symbol: str, length: int, multiplier: float, timeframe: str):
        
        #return sell/buy

    def run(self):
        logger = logging.getLogger()
        for symbol in self.symbols:
            day_super_trend_signal_10_3 = self.getSuperTrend(symbol, 10, 3, self.data_timeframe)
            day_super_trend_signal_1_1 = self.getSuperTrend(symbol, 1, 1, self.data_timeframe)

            logger.info(f"for {symbol} / 10, 3 is: {day_super_trend_signal_10_3}")
            logger.info(f"for {symbol} / 1, 1 is: {day_super_trend_signal_1_1}")

bot = SuperTrend()
bot.run()

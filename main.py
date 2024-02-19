### DEPENDENCIES ###
from datetime import date
import alpaca_trade_api as api
import numpy as np 
import pandas as pd
import time
import warnings
import threading
import math
import config
import logging
warnings.filterwarnings('ignore')

class StockBot:
    def __init__(self):
        self.alpaca = api.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_URL, "v2")
        self.account = self.alpaca.get_account()

        self.sleeper = config.ALPACA_SLEEP_TIMEOUT
        self.symbols = config.ALPACA_STOCK_CONFIG
        self.bot_version = config.ALPACA_STOCKING_BOT_VERSION

        self.data_timeframe = "1Day"

        log_format = "%(levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(filename = "debug.log", filemode = "w", format = log_format, level = logging.DEBUG)

    def run(self):
        logger = logging.getLogger()

        logger.debug(f"Stocking Bot v{self.bot_version}. Running.... ---")
        logger.debug("I am checking if market is open.... ---")
        marketOpenThread = threading.Thread(target=self.isMarketOpen)
        marketOpenThread.start()
        marketOpenThread.join()
        logger.debug("The market is open. Good luck! ---")

        logger.debug("Time to clean. I am starting close all positions!... ---")
        self.cleaningTrade()
        logger.debug("Clening done..... ---")

        logger.debug("I am running first trading cycle.... ---")
        tradeBotThread = threading.Thread(target=self.tradeBot)
        tradeBotThread.start()
        tradeBotThread.join()
        logger.debug("First trading cycle done. Sleeping.... ---")

    def cleaningTrade(self):
        logger = logging.getLogger()        
        try:
            portfolio = self.alpaca.list_positions()
            for position in portfolio:
                day_super_trend_signal_10_03 = self.getSuperTrend(position.symbol, 10, 3, self.data_timeframe)
                day_super_trend_signal_01_01 = self.getSuperTrend(position.symbol, 1, 1, self.data_timeframe)
                #
                if day_super_trend_signal_10_03 == 'hold' or day_super_trend_signal_01_01 == 'hold':
                    logger.debug(f"I am trying to close all positions for the {position.symbol}. ---")
                    self.alpaca.close_position(position.symbol)
                    time.sleep(1)
                    logger.debug(f"All positions on the {position.symbol} are closed successfully! ---")
                else:
                    logger.debug(f"The order for close by {position.symbol} was skipping! ---")
        except Exception as e:
            logger.error(f"I can't close positions for a reason: {e} ---")  

    def isMarketOpen(self):
        logger = logging.getLogger()
        isOpen = self.alpaca.get_clock().is_open
        while(not isOpen):
            next_open_time = self.alpaca.get_clock().next_open.timestamp()
            current_time = self.alpaca.get_clock().timestamp.timestamp()
            difference = (int(next_open_time) - int(current_time))/60
            hours = round(difference // 60)
            remaining_minutes = round(difference % 60)
            logger.debug(f"The {hours} hours and {remaining_minutes} minutes until the next market trade session! ---")
            logger.debug(f"I am sleeping for {hours} hours and {remaining_minutes} minutes. Good night! :) ---")
            time.sleep((difference+15)*60)
            isOpen = self.alpaca.get_clock().is_open
            
    def submitOrder(self, symbol, quantity, side, price):
        logger = logging.getLogger()

        low_price   = round((price * 0.99), 2) # SL 01%
        hight_price = round((price * 1.05), 2) # TP 05%

        logger.debug(f"The take profit for {symbol} is {hight_price}---")
        logger.debug(f"The stop loss for {symbol} is {low_price}---")
        logger.debug(f"Signal: {side} ---")

        try:
            self.alpaca.submit_order(
                symbol = symbol,
                qty = quantity,
                side = side,
                type = 'market',
                time_in_force = 'day',
                order_class = 'bracket',
                stop_loss={
                    'stop_price': low_price
                },
                take_profit={
                    'limit_price': hight_price
                }
            )             
            logger.debug(f"The order for {symbol} for {quantity} shares @ {price} has been submitted! ---")
        except Exception as e:
            logger.error(e)
            logger.error(f"The order for {symbol} for {quantity} shares @ {price} has NOT been submitted! ---")

    def symbolPositionBySymbol(self, symbol):
        logger = logging.getLogger()
        try:
            position = self.alpaca.get_position(symbol)
            return position
        except:
            return

    def symbolPositionByAssetID(self, asset_id):
        logger = logging.getLogger()
        try:
            position = self.alpaca.get_position(asset_id)
            return position
        except:
            return 
            
    def tradeBot(self):
        logger = logging.getLogger()

        quoteList = []
        
        try:
            if self.account.trading_blocked:
                logger.debug(f"Account is currently restricted from trading! ---")
            else:
                logger.debug(f"Account status is: {self.account.status}. ---")
                cash = float(self.account.cash)
                logger.debug(f"Account cash is: {cash}. ---")
                for symbol in self.symbols:
                    price = self.alpaca.get_latest_quote(symbol).ap 
                    time.sleep(1)

                    if price > 0:   
                        logger.debug(f"Current price: {price}. ---")

                        percent_trade = float(self.symbols[symbol])
                        logger.debug(f"Percent trade: {percent_trade}. ---")

                        limit_trade = (cash * (percent_trade / 100))
                        logger.debug(f"Limit trade : {limit_trade}. ---")

                        quantity = 0

                        if self.symbolPositionBySymbol(symbol):
                            position = self.symbolPositionBySymbol(symbol)
                            market_value = abs(float(position.market_value))
                            logger.debug(f"Market value : {market_value}. ---")
                            free_trade = limit_trade - market_value
                            quantity = (round(free_trade / price))
                            logger.debug(f"Free trade : {free_trade}. ---")
                        else:
                            quantity = (round(limit_trade / price))
                            logger.warning(f"I don't have position by {symbol}! ---")
                        
                        day_super_trend_signal_10_03 = self.getSuperTrend(symbol, 10, 3, self.data_timeframe)
                        day_super_trend_signal_01_01 = self.getSuperTrend(symbol, 1, 1, self.data_timeframe)

                        logger.debug(f"<<< ST Signal: {day_super_trend_signal_10_03} for {symbol}... >>>")
                        logger.debug(f"<<< ST Signal: {day_super_trend_signal_01_01} for {symbol}... >>>")

                        if day_super_trend_signal_10_03 == 'hold' or day_super_trend_signal_01_01 == 'hold':
                            logger.warning(f"The sp500 super trend indicator has hold signal for {symbol}...skipping! ---")
                            quantity = 0
                        else:
                            if quantity > 1:
                                submitOrderThread = threading.Thread(target = self.submitOrder, args=[symbol, quantity, "buy", price])
                                submitOrderThread.start()
                                submitOrderThread.join()
                            else:
                                logger.warning(f"I am sorry. All authorized funds for the {symbol} are currently occupied for trading...skipping! ---")
                    else:
                        logger.warning(f"I can't get the price per {symbol} at the moment...skipping! ---")
        except Exception as e:
            logger.debug(e)

    def getSuperTrend(self, symbol: str, length: int, factor: int, timeframe: str):
        # Get the 1-day bars for the symbol
        data_bars = self.alpaca.get_bars(symbol, timeframe, limit=length)
        
        # Calculate the super trend indicator
        atr = (data_bars[0].h - data_bars[0].l) / data_bars[0].c
        super_trend = (data_bars[0].h + data_bars[0].l) / 2 + length * atr * factor
        
        if data_bars:
            if super_trend > data_bars[0].c:
                return 'buy'
            elif super_trend < data_bars[0].c:
                return 'hold'
            else:
                return 'hold'
        else:
            logger.error(f'Data Bars: {data_bars}!')

    def botLoop(self):
        bot = StockBot()
        while True:
            bot.run()
            time.sleep(self.sleeper * 60)

bot = StockBot()
bot.botLoop()
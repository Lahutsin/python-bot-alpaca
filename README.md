# Day Trader (Index Bot)

## Configuration:

```
# LIVE
#ALPACA_KEY          = "YOUR ALPACA KEY"
#ALPACA_SECRET_KEY   = "YOUR ALPACA SECRET KEY"
#ALPACA_URL          = "https://api.alpaca.markets"

# TEST
ALPACA_KEY           = ""
ALPACA_SECRET_KEY    = ""
ALPACA_URL           = "https://paper-api.alpaca.markets"

# STOCK LIST
# Example: NAME_STOCK : PERCENT (%)
ALPACA_STOCK_CONFIG  = {
    'TQQQ' : 10.3,
    'AAPL' : 10.3
}

# TIME TICKERS
ALPACA_SLEEP_TIMEOUT = 360 # 6H

# INFO
ALPACA_STOCKING_BOT_VERSION = "0.1.7a"
```
## 
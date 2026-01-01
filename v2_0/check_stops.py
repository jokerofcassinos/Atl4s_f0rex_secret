import MetaTrader5 as mt5
import config

if not mt5.initialize():
    print(f"Failed to init MT5: {mt5.last_error()}")
else:
    info = mt5.symbol_info(config.SYMBOL)
    if info:
        print(f"Symbol: {info.name}")
        print(f"Digits: {info.digits}")
        print(f"Point: {info.point}")
        print(f"StopLevel: {info.trade_stops_level}")
        print(f"Spread: {info.spread}")
        print(f"Min Distance (Points): {info.trade_stops_level}")
        print(f"Min Distance (Price): {info.trade_stops_level * info.point}")
    else:
        print(f"Symbol {config.SYMBOL} not found")
    mt5.shutdown()

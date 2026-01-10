import pandas as pd
import logging

logger = logging.getLogger("TimeConverter")

class TimeframeConverter:
    """
    Converts base timeframe data (e.g., M1) into custom timeframes.
    Special focus on Fibonacci Timeframes (e.g., 8 minutes).
    """

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, timeframe_minutes: int, timeframe_label: str = None) -> pd.DataFrame:
        """
        Resamples OHLCV DataFrame to a specific minute interval.
        
        Args:
            df: DataFrame with M1 data (must have DatetimeIndex).
            timeframe_minutes: The target interval in minutes (e.g., 8).
            timeframe_label: Label for logging/naming.
            
        Returns:
            Resampled DataFrame with Open, High, Low, Close, Volume.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for resampling.")
            return pd.DataFrame()

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                # Attempt to find a time column or convert index
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.error(f"Failed to convert index to DatetimeIndex: {e}")
                return pd.DataFrame()

        # Define resampling rule
        rule = f'{timeframe_minutes}T' # 'T' stands for minutes in pandas legacy, 'min' is newer but T works universally

        # Resample logic
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Handle 'tic' volume or other columns if present
        if 'tick_volume' in df.columns:
            agg_dict['tick_volume'] = 'sum'
        
        try:
            resampled = df.resample(rule, label='right', closed='right').agg(agg_dict)
            
            # Drop NaN rows (gaps)
            resampled.dropna(inplace=True)
            
            if timeframe_label:
                logger.info(f"Resampled {len(df)} M1 rows to {len(resampled)} {timeframe_label} rows.")
                
            return resampled

        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return pd.DataFrame()

    @staticmethod
    def generate_fibonacci_timeframes(df_m1: pd.DataFrame) -> dict:
        """
        Generates a suite of Fibonacci timeframes from M1 data.
        Sequence: 1, 2, 3, 5, 8, 13, 21, 34, 55...
        """
        fib_sequence = [2, 3, 5, 8, 13, 21, 34, 55]
        timeframes = {'M1': df_m1}
        
        for m in fib_sequence:
            label = f"M{m}"
            timeframes[label] = TimeframeConverter.resample_ohlcv(df_m1, m, label)
            
        return timeframes

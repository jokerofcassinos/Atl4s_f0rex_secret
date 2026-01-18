import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugLoader")

def load_csv_data(file_path, start_date=None, end_date=None):
    logger.info(f"Loading data from {file_path}")
    try:
        # Initial read to detect format (Optimized for speed)
        df = pd.read_csv(file_path, nrows=5)
        
        # Check formatting
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep='\t', parse_dates=False)
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=';', parse_dates=False)
        else:
             df = pd.read_csv(file_path, parse_dates=False) # Reload full
        
        # CHECK FOR HEADERLESS CSV
        first_col = str(df.columns[0])
        if first_col.startswith(('20', '19')) and len(first_col) >= 4:
            logger.info("Detected Headerless CSV (First row is data). Reloading with header=None...")
            # Reload with correct separator
            sep = ','
            if len(df.columns) == 1: sep = '\t' 
            
            df = pd.read_csv(file_path, header=None, parse_dates=False)
            
            # Assign default MT5 headers based on column count
            if len(df.columns) >= 7:
                 cols = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VOL_REAL', 'SPREAD']
                 df.columns = cols[:len(df.columns)]
                 logger.info(f"Assigned Default Headers: {list(df.columns)}")

        # Normalize Columns
        df.columns = [c.upper().strip() for c in df.columns]
        logger.info(f"Loaded Columns: {list(df.columns)}")
        
        rename_map = {
            '<DATE>': 'DATE', '<TIME>': 'TIME', '<OPEN>': 'OPEN', '<HIGH>': 'HIGH', '<LOW>': 'LOW', '<CLOSE>': 'CLOSE', '<TICKVOL>': 'VOLUME', '<VOL>': 'VOLUME',
            'DATE': 'DATE', 'TIME': 'TIME', 'OPEN': 'OPEN', 'HIGH': 'HIGH', 'LOW': 'LOW', 'CLOSE': 'CLOSE', 'VOLUME': 'VOLUME', 'VOL': 'VOLUME'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Combine Date and Time
        if 'DATE' in df.columns and 'TIME' in df.columns:
            logger.info("Parsing Date and Time columns...")
            try:
                # Optimized combining
                df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
            except:
                logger.info("Fast parsing failed, trying format...")
                df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), format='%Y.%m.%d %H:%M:%S', errors='coerce')
            
            df.set_index('datetime', inplace=True)
            df.drop(columns=['DATE', 'TIME'], inplace=True)
            
        elif 'DATETIME' in df.columns:
             df['datetime'] = pd.to_datetime(df['DATETIME'])
             df.set_index('datetime', inplace=True)

        # Rename for system compatibility (lowercase)
        df.columns = [c.lower() for c in df.columns]

        # FILTER DATE RANGE (Optimized before resampling)
        if start_date:
            logger.info(f"Filtering Start: {start_date}")
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            logger.info(f"Filtering End: {end_date}")
            df = df[df.index <= pd.Timestamp(end_date)]
            
        return df

    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    file = r"D:\Atl4s-Forex\historical_datas\DAT_MT_GBPUSD_M1_2016.csv"
    start = "2016-10-06"
    end = "2016-10-08"
    
    print(f"Testing Loader on {file} for Range {start} to {end}")
    df = load_csv_data(file, start, end)
    
    if df is not None:
        print("\n--- RESULTS ---")
        print(f"Rows: {len(df)}")
        if not df.empty:
            print("\nFirst 5 Rows:")
            print(df.head())
            print("\nLast 5 Rows:")
            print(df.tail())
            
            print(f"\nIndex Type: {df.index.dtype}")
            print(f"Price at Index 0: {df.iloc[0]['open']}")
        else:
            print("DataFrame is Empty!")

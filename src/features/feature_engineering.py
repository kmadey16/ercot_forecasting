import pandas as pd
import numpy as np
import os
import datetime as dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'


os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    df = pd.read_parquet(INTERIM_DIR / 'merged_all_data.parquet')
    return df


# Need to fix the data quality issues found in EDA
def fix_data_quality(df):

    # Drop duplicate timestamps
    df = df.drop_duplicates(subset='timestamp', keep='first')

    # Keep TotalResourceMWZoneWest — West zone generator outage capacity is a direct
    # congestion signal (less local generation → more imports → higher local prices).
    # Previously dropped; reinstated for the price model. ~3,500 nulls in early 2021
    # when zone-level data wasn't published — left as NaN so LGBM handles natively
    # (it can split on missing values). Do NOT dropna or ffill — preserves training rows.

    # Cap wind solar to fix bad data -- Limit to ERCOTs installed capacity
    df.loc[df['WGRPP_SYSTEM_WIDE'] > 45000, 'WGRPP_SYSTEM_WIDE'] = np.nan
    df.loc[df['PVGRPP_SYSTEM_WIDE'] > 25000, 'PVGRPP_SYSTEM_WIDE'] = np.nan
    df.loc[df['WGRPP_LZ_WEST'] > 25000, 'WGRPP_LZ_WEST'] = np.nan

    # Forward fill the small gaps for prices and load (the few nulls we saw in EDA)
    df[['RT_price','DAM_price' ,'hub_load', 'load_total','COP_HSL_SYSTEM_WIDE']] = df[['RT_price','DAM_price' ,'hub_load', 'load_total','COP_HSL_SYSTEM_WIDE']].ffill().bfill()

    # Drop trivial nulls (5-20 rows) 
    df = df.dropna(subset=['PVGRPP_SYSTEM_WIDE', 'WGRPP_LZ_WEST', 'PRC']).copy()

    #Recalc DA RT Spread
    df['RT_DAM_spread'] = df['RT_price'] - df['DAM_price']

    # DAM congestion spread: how much more expensive HB_WEST clears vs system average.
    # Positive = DAM already expects congestion into West Texas. This is known 24h ahead
    # (DAM clears day-ahead) and is the most direct forward-looking congestion signal.
    if 'DAM_system' in df.columns:
        df['DAM_congestion_spread'] = df['DAM_price'] - df['DAM_system']
    else:
        df['DAM_congestion_spread'] = np.nan

    return df

# Feature building - time based (hour,day,month,weekend,cyclic)

def add_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.day_of_week.isin([5,6]).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # RTC+B market redesign flag (Dec 5, 2025). Pre-RTC+B PRC and adder data
    # come from ERCOT xlsx archives; post comes from GridStatus.io API.
    # The flag lets the model learn any systematic level shift between the two regimes.
    df['is_pre_rtcb'] = (df['timestamp'] < pd.Timestamp('2025-12-05')).astype(int)

    return df

# Lagged features (1h, 24h, 168h lags for RT, DAM, 24h for load and 24/268 for wind)
def add_lag_features(df):
    
    #prices
    df['RT_1h_lag'] = df['RT_price'].shift(1)
    df['RT_24h_lag'] = df['RT_price'].shift(24)
    df['RT_168h_lag'] = df['RT_price'].shift(168)

    df['DAM_1h_lag'] = df['DAM_price'].shift(1)
    df['DAM_24h_lag'] = df['DAM_price'].shift(24)
    df['DAM_168h_lag'] = df['DAM_price'].shift(168)

    #load
    df['hub_load_24h_lag'] = df['hub_load'].shift(24)
    df['load_total_24h_lag'] = df['load_total'].shift(24)

    #wind
    df['wind_24h_lag'] = df['WGRPP_LZ_WEST'].shift(24)
    df['wind_168h_lag'] = df['WGRPP_LZ_WEST'].shift(168)

    # PRC lags — discovered during modeling as the single biggest improvement (65% MAE reduction)
    df['PRC_1h_lag'] = df['PRC'].shift(1)
    df['PRC_24h_lag'] = df['PRC'].shift(24)
    df['PRC_168h_lag'] = df['PRC'].shift(168)

    # RT-DAM spread lag — when RT exceeds DAM, the market is under more stress than
    # day-ahead expected. Lagged 1h so it's observable at prediction time. This is the
    # key "market surprise" signal for the price model.
    df['RT_DAM_spread_1h_lag'] = df['RT_DAM_spread'].shift(1)

    # Price spike lag — binary flag: was the previous hour's RT price > $100/MWh?
    # Spikes cluster in time (congestion events persist across hours), so knowing the
    # prior hour spiked is a strong predictor of the current hour spiking too.
    df['price_spike_lag'] = (df['RT_price'].shift(1) > 100).astype(int)

    # PRC rolling stats — capture decline trajectory, not just point-in-time state.
    # A genuine scarcity onset has PRC declining over several hours; brief dips recover quickly.
    df['PRC_roll_mean_6h'] = df['PRC'].shift(1).rolling(6).mean()
    df['PRC_roll_std_6h'] = df['PRC'].shift(1).rolling(6).std()
    df['PRC_roll_mean_24h'] = df['PRC'].shift(1).rolling(24).mean()

    return df

# Rolling stats of lagged prices/load/wind (these cols because they are volatile with repeating patterns)
def add_rolling_stats(df):
    for col in ['RT_price', 'DAM_price', 'hub_load', 'load_total', 'WGRPP_LZ_WEST']:
        df[f'{col}_roll_mean_6h'] = df[col].shift(1).rolling(6).mean()
        df[f'{col}_roll_std_6h'] = df[col].shift(1).rolling(6).std()
        df[f'{col}_roll_mean_24h'] = df[col].shift(1).rolling(24).mean()
        df[f'{col}_roll_std_24h'] = df[col].shift(1).rolling(24).std()

    return df

# Add in engineered features (net load, renewable penetration, ramp rates)

def add_engineered_features(df):

    # Add renewable forecast error
    df['wind_forecast_error'] = df['WGRPP_LZ_WEST'] - df['STWPF_LZ_WEST']
    df['solar_forecast_error'] = df['PVGRPP_SYSTEM_WIDE'] - df['STPPF_SYSTEM_WIDE']

    #Net load = demand after renewables (two versions due to data limits)
    df['net_load_system'] = df['load_total'] - df['WGRPP_SYSTEM_WIDE'] - df['PVGRPP_SYSTEM_WIDE']
    df['net_load_west'] = df['hub_load'] - df['WGRPP_LZ_WEST'] #No data for WEST hub solar

    # Renewable penetration -system wide and WEST-
    df['renewable_pct_system'] = (df['WGRPP_SYSTEM_WIDE'] + df['PVGRPP_SYSTEM_WIDE']) / df['load_total']
    df['renewable_pct_west'] = df['WGRPP_LZ_WEST'] / df['hub_load']

    # Wind-to-load ratio for West Texas — captures local supply/demand balance.
    # When wind drops relative to load, the zone relies more on imports through
    # congested transmission lines, driving local price spikes even when system-wide
    # reserves (PRC) are comfortable. This is the core congestion signal.
    df['wind_load_ratio_west'] = df['WGRPP_LZ_WEST'] / df['hub_load']

    #Track increasing PRC over time
    df['days_since_start'] = (df['timestamp'] - pd.Timestamp('2021-01-01')).dt.days

    #Ramp rates (track hour over hour changes)
    df['load_total_ramp'] = df['load_total'].diff()
    df['hub_load_ramp'] = df['hub_load'].diff()
    df['wind_west_ramp'] = df['WGRPP_LZ_WEST'].diff()
    df['wind_system_ramp'] = df['WGRPP_SYSTEM_WIDE'].diff()
    df['RT_price_ramp'] = df['RT_price'].diff()

    return df

def add_ordc_features(df):
    """
    Log1p-transform ORDC price adder columns from the merged dataset.

    RTORPA is extremely right-skewed: 84% zeros, typical range 0-10 $/MWh,
    but Uri 2021 hit ~4,143. Raw values would let that single event dominate
    linear model coefficients. log1p compresses the spike (4143 -> 8.3) while
    keeping zero as zero and preserving the relative ordering.

    Post-RTC+B rows (is_pre_rtcb=0) are NaN until gridstatus adder data is
    pulled — log1p(NaN)=NaN, which propagates intentionally. LGBM handles
    NaN natively; LR will need imputation at retrain time.
    """
    for col in ['RTORPA', 'RTOFFPA', 'RTORDPA']:
        df[f'{col}_log1p'] = np.log1p(df[col])
    return df


# Add regime labels
def add_regime_labels(df):
    bins = [0, 3000, 5000, 10000, float('inf')]
    labels = ['Scarcity', 'Tight', 'Normal', 'Surplus']
    df['regime'] = pd.cut(df['PRC'], bins=bins, labels=labels)
    return df


def feature_engineering_pipeline():
    df = load_data()
    df = fix_data_quality(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_engineered_features(df)
    df = add_ordc_features(df)
    df = add_regime_labels(df)

    # Drop lag warmup rows (first 168 hours)
    df = df.iloc[168:].reset_index(drop=True)

    df.to_parquet(PROCESSED_DIR / 'model_ready.parquet')

    return df

if __name__ == '__main__':
    feature_engineering_pipeline()
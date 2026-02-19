import pandas as pd
import numpy as np
import os
import datetime as dt

PROJECT_ROOT = '/Users/kamilmadey/Desktop/ercot_forecasting_project/'
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    df = pd.read_parquet(INTERIM_DIR + '/merged_all_data.parquet')
    return df


# Need to fix the data quality issues found in EDA
def fix_data_quality(df):
    for col in ['WGRPP_LZ_WEST', 'PVGRPP_SYSTEM_WIDE']:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # Drop duplicate timestamps
    df = df.drop_duplicates(subset='timestamp', keep='first')

    # Forward fill the small gaps for prices and load (the few nulls we saw in EDA)
    df[['RT_price','DAM_price' ,'hub_load', 'load_total']] = df[['RT_price','DAM_price' ,'hub_load', 'load_total']].ffill()

    # Backfill Outage nulls for West (For non tree based models) 
    df['TotalResourceMWZoneWest'] = df['TotalResourceMWZoneWest'].bfill()

    # Drop trivial nulls (5-20 rows)
    df = df.dropna(subset=['PVGRPP_SYSTEM_WIDE', 'WGRPP_LZ_WEST', 'West']).copy()

    #Recalc DA RT Spread
    df['RT_DAM_spread'] = df['RT_price'] - df['DAM_price']

    return df

# Feature building - time based (hour,day,month,weekend,cyclic)

def add_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.day_of_week.isin([5,6]).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

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
    
    #Net load = demand after renewables (two versions due to data limits)
    df['net_load_system'] = df['load_total'] - df['WGRPP_SYSTEM_WIDE'] - df['PVGRPP_SYSTEM_WIDE']
    df['net_load_west'] = df['hub_load'] - df['WGRPP_LZ_WEST'] #No data for WEST hub solar

    # Renewable penetration -system wide and WEST-
    df['renewable_pct_system'] = (df['WGRPP_SYSTEM_WIDE'] + df['PVGRPP_SYSTEM_WIDE']) / df['load_total']
    df['renewable_pct_west'] = df['WGRPP_LZ_WEST'] / df['hub_load']

    #Ramp rates (track hour over hour changes)
    df['load_total_ramp'] = df['load_total'].diff()
    df['hub_load_ramp'] = df['hub_load'].diff()
    df['wind_west_ramp'] = df['WGRPP_LZ_WEST'].diff()
    df['wind_system_ramp'] = df['WGRPP_SYSTEM_WIDE'].diff()
    df['RT_price_ramp'] = df['RT_price'].diff()

    return df

# Add regime labels
def add_regime_labels(df):
    regime_conditions = [df['RT_price'] <= 0, (df['RT_price'] > 0) & (df['RT_price'] <= 75), (df['RT_price'] > 75) & (df['RT_price'] <= 200), df['RT_price'] > 200]
    labels = ['Low', 'Normal', 'Tight', 'Scarcity']

    df['regime'] = np.select(regime_conditions,labels, default='Unknown')

    return df


def feature_engineering_pipeline():
    df = load_data()
    df = fix_data_quality(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_engineered_features(df)
    df = add_regime_labels(df)

    # Drop nulls from lagged features
    df = df.dropna().reset_index(drop=True)

    df.to_parquet(PROCESSED_DIR + '/model_ready.parquet')

    return df

if __name__ == '__main__':
    feature_engineering_pipeline()
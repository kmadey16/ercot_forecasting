# src/data/preprocess.py

"""
Data preprocessing: cleaning and merging
Takes raw data → produces interim data

Handles:
- RT LMP data (15-min intervals)
- DAM LMP data (hourly)
- System load data
- Weather data (API)
"""

import pandas as pd
import numpy as np
import requests
import os
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
base = BASE_DIR / 'data' / 'raw'
interim = BASE_DIR / 'data' / 'interim'

def fix_hour24_helper(df, date_col, hour_col):
    df[hour_col] = df[hour_col].astype(str).str.strip().str.replace(':00', '')
    df[date_col] = pd.to_datetime(df[date_col])  
    
    mask = df[hour_col] == '24'
    df.loc[mask, date_col] = df.loc[mask, date_col] + pd.Timedelta(days=1)
    df.loc[mask, hour_col] = '0'
    
    df['timestamp'] = df[date_col] + pd.to_timedelta(df[hour_col].astype(int), unit='h')
    return df

def clean_RT_data(filepaths, hub='HB_WEST'):
    """
    Clean RT LMP data from raw Selenium-scraped files
    
    Parameters:
    -----------
    filepaths : list of str
        List of paths to RT LMP CSV files
        Example: ['file1.csv', 'file2.csv', 'file3.csv']
    hub : str
        Hub to filter to (default: 'HB_WEST')
    
    Returns:
    --------
    pd.DataFrame
        Columns: ['timestamp', 'RT_price']
    """

    dfs = []
    for i, filepath in enumerate(filepaths, 1):
        print(f"  [{i}/{len(filepaths)}] Loading: {filepath}")
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    # Combine
    df_final = pd.concat(dfs, ignore_index=True)

    #filter to hub
    df_west = df_final[df_final["SettlementPointName"] == hub].copy()

    # Add timestamp column
    df_west['DeliveryDate'] = pd.to_datetime(df_west['DeliveryDate'])
    df_west['timestamp'] = df_west['DeliveryDate'] + pd.to_timedelta((df_west['DeliveryHour'] - 1) * 60 + (df_west['DeliveryInterval'] - 1) * 15, unit='m')

    # Aggregate 15-min to hourly
    df_west['hour'] = df_west['timestamp'].dt.floor('h')
    df_rt_hourly = df_west.groupby('hour').agg({
        'SettlementPointPrice': 'mean'
    }).reset_index()
    df_rt_hourly.columns = ['timestamp', 'RT_price']

    df_rt_hourly = df_rt_hourly.sort_values('timestamp').reset_index(drop=True)
    
    return df_rt_hourly

def clean_DAM_data(filepaths, hub='HB_WEST'):
    """
    Clean DAM (Day-Ahead Market) LMP data from raw files
    - Combines any number of year files
    - Filters to specified hub
    - Handles hour 24 (midnight next day)
    - Standardizes timestamp to hourly
    
    Parameters:
    -----------
    *filepaths : str
        Paths to DAM LMP CSV files (can pass any number)
        Example: clean_dam_lmp_data('file1.csv', 'file2.csv', 'file3.csv')
    hub : str
        Hub to filter to (default: 'HB_WEST')
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DAM LMP data with columns: ['timestamp', 'DAM_price']
        
    Output Columns:
    ---------------
    - timestamp : datetime64
        Hourly timestamp (handles hour 24 correctly)
    - dam_lmp : float
        Day-ahead locational marginal price ($/MWh)

    """
    dfs = []
    for i, filepath in enumerate(filepaths, 1):
        print(f"  [{i}/{len(filepaths)}] Loading: {filepath}")
        df = pd.read_csv(filepath)
        dfs.append(df)
    # Combine
    df_final = pd.concat(dfs, ignore_index=True)

    #Filter hub
    df_da = df_final[df_final["SettlementPoint"] == hub].copy()

    # Fix HourEnding col
    df_da = fix_hour24_helper(df_da, 'DeliveryDate', 'HourEnding')
    

    # Clean up - just keep timestamp and price
    df_da_clean = df_da[['timestamp', 'SettlementPointPrice']].copy()
    df_da_clean.columns = ['timestamp', 'DAM_price']
    df_da_clean = df_da_clean.drop_duplicates(subset='timestamp', keep='first')

    return df_da_clean

def clean_prc(pre_rtcb_dir, post_rtcb_filepath):
    """
    Build unified PRC series from pre and post RTC+B sources.
    Pre: ERCOT archive xlsx files (2021 - Dec 4, 2025)
    Post: GridStatus.io CSV/parquet (Dec 5, 2025+)
    """
    # 1. Read and process all pre-RTC+B xlsx files
    
    all_years = []
    for filepath in sorted(Path(pre_rtcb_dir).glob("price_adders_*.xlsx")):
        sheet_dict = pd.read_excel(filepath, sheet_name=None)
        # your sheet loop here
        all_sheets =[]
        for name, sheet in sheet_dict.items():
            if name.lower() in ['report info', 'sheet1']:
                continue
            sheet.columns = sheet.iloc[7].values
            sheet = sheet.iloc[8:].reset_index(drop=True)
            sheet.columns = sheet.columns.str.strip(" ")
    
            cols = ['SCED Timestamp', 'PRC']
            sheet = sheet[cols].copy()
            
            sheet['SCED Timestamp'] = pd.to_datetime(sheet['SCED Timestamp'])
            sheet['hour'] = sheet['SCED Timestamp'].dt.floor('h')

            sheet = sheet.groupby('hour').agg({'PRC': "min"}).reset_index()

            all_sheets.append(sheet)

        full_table = pd.concat(all_sheets)
        full_table.reset_index(inplace=True, drop=True)
        all_years.append(full_table)
    
    pre = pd.concat(all_years).reset_index(drop=True)
    pre['PRC'] = pd.to_numeric(pre['PRC'])
    pre.loc[pre['PRC'] < 0, 'PRC'] = np.nan
    pre['PRC'] = pre['PRC'].interpolate()
    pre = pre[['hour', 'PRC']]
    pre = pre.rename(columns={'hour': 'timestamp'})

    # 2. Read post-RTC+B
    post = pd.read_csv(post_rtcb_filepath)
    post['timestamp'] = pd.to_datetime(post['interval_start_local']).dt.tz_localize(None)
    post = post[['timestamp', 'prc']].rename(columns={'prc':'PRC'})

    # 3. Concat
    prc = pd.concat([pre, post]).sort_values('timestamp').reset_index(drop=True)
    
    return prc

def clean_load_data(filepaths, hub = 'WEST'):
    
    
    dfs = []
    for i, filepath in enumerate(filepaths, 1):
        print(f"  [{i}/{len(filepaths)}] Loading: {filepath}")
        df = pd.read_csv(filepath)
        dfs.append(df)

    # Combine
    load_df = pd.concat(dfs, ignore_index=True)
    
    # Clean hour col and add timestmap
    load_df = fix_hour24_helper(load_df, 'OperDay', 'HourEnding')

    #Filter cols
    df_load_clean = load_df[['timestamp', hub, 'TOTAL']]
    df_load_clean.columns = ['timestamp', 'hub_load', 'load_total']
    df_load_clean = df_load_clean.drop_duplicates(subset='timestamp', keep='first')

    return df_load_clean

def clean_wind_data(filepaths, hub = 'WEST'):
    
    dfs= []
    for i, filepath in enumerate(filepaths, 1):
        df = pd.read_csv(filepath)
        dfs.append(df)
    #combine
    wind_df = pd.concat(dfs,ignore_index=True)

    # Filter to keep only most recent forecast
    wind_df = wind_df.sort_values(by='source_file').drop_duplicates(subset=['DELIVERY_DATE', 'HOUR_ENDING'],keep='last')
    
    #Fix Hour-ending and add timestamp
    wind_df = fix_hour24_helper(wind_df, 'DELIVERY_DATE', 'HOUR_ENDING')

    # filter cols we will use --- the non-nulls
    wind_df = wind_df[['timestamp','WGRPP_LZ_{}'.format(hub),'STWPF_LZ_{}'.format(hub),'COP_HSL_LZ_{}'.format(hub),'WGRPP_SYSTEM_WIDE']]

    return wind_df


def clean_solar(filepaths):
    dfs= []
    for i, filepath in enumerate(filepaths, 1):
        df = pd.read_csv(filepath)
        dfs.append(df)
    #combine
    solar_df = pd.concat(dfs,ignore_index=True)
    
    #Keep only most recent forecast
    solar_df = solar_df.sort_values(by='source_file').drop_duplicates(subset=['DELIVERY_DATE', 'HOUR_ENDING'],keep='last')
    
    #Fix Hour-ending
    solar_df = fix_hour24_helper(solar_df, 'DELIVERY_DATE', 'HOUR_ENDING')

    # Isolate the cols we need
    solar_df = solar_df[['timestamp', 'PVGRPP_SYSTEM_WIDE', 'STPPF_SYSTEM_WIDE', 'COP_HSL_SYSTEM_WIDE']]

    return solar_df
    
def clean_fcst_data(filepaths):
    
    dfs= []
    for i, filepath in enumerate(filepaths, 1):
        df = pd.read_csv(filepath)
        dfs.append(df)
    #combine
    fcst_df = pd.concat(dfs,ignore_index=True)
    
    # only take the model used fcst
    fcst_df = fcst_df[fcst_df['InUseFlag'] == 'Y'].copy()

    # Fix Hour ending
    fcst_df = fix_hour24_helper(fcst_df, 'DeliveryDate', 'HourEnding')

    #filter df
    fcst_df = fcst_df[['timestamp', 'West', 'SystemTotal']]
    fcst_df = fcst_df.rename(columns={'West': 'load_fcst_west', 'SystemTotal': 'load_fcst_system'})
    fcst_df = fcst_df.drop_duplicates(subset='timestamp',keep='first')

    return fcst_df

def clean_outages_data(filepaths):

    dfs = []
    for i , filepath in enumerate(filepaths, 1):
        df = pd.read_csv(filepath)
        dfs.append(df)

    outages_df = pd.concat(dfs,ignore_index=True)

    #Fix hour ending
    outages_df = fix_hour24_helper(outages_df, 'Date', 'HourEnding')
    outages_df = outages_df.drop_duplicates(subset='timestamp', keep='first')

    # Zones
    zone_cols = ['TotalResourceMWZoneSouth', 'TotalResourceMWZoneNorth','TotalResourceMWZoneWest', 'TotalResourceMWZoneHouston']

    # Sum Zone cols to get total (Total not available for entire dataset)
    mask = outages_df[zone_cols].notna().all(axis=1)
    outages_df.loc[mask, 'TotalResourceMW_calc'] = outages_df.loc[mask, zone_cols].sum(axis=1)

    # Use raw TotalResourceMW for early 2021, calculated for the rest
    outages_df['TotalResourceMW_final'] = outages_df['TotalResourceMW_calc'].fillna(outages_df['TotalResourceMW'])

    # Filter df
    outages_df = outages_df[['timestamp', 'TotalResourceMWZoneWest', 'TotalResourceMW_final']]
    outages_df.columns = ['timestamp', 'TotalResourceMWZoneWest', 'TotalResourceMW']

    return outages_df

def fetch_weather_data(start_date='2021-01-01', end_date='2025-12-31', 
                       lat=31.99, lon=-102.08, location_name='Midland, TX'):
    """
    Fetch weather data from Open-Meteo API
    - Uses Midland, Texas coordinates (center of West Texas HUB)
    - Hourly weather data
    
    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    lat : float
        Latitude (default: 31.99 for Midland, TX)
    lon : float
        Longitude (default: -102.08 for Midland, TX)
    location_name : str
        Location name for display
    
    Returns:
    --------
    pd.DataFrame
        Weather data with columns: ['timestamp', 'temperature', 'humidity', 'windspeed', 'precipitation']
    """
    
    # Open-Meteo API endpoint for historical data
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation"],
        "timezone": "America/Chicago"  # CST/CDT for Texas
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f" API request failed with status code {response.status_code}")
        return None
    
    weather_data = response.json()
    
    # Convert to DataFrame
    weather_df = pd.DataFrame({
        'timestamp': pd.to_datetime(weather_data['hourly']['time']),
        'temperature': weather_data['hourly']['temperature_2m'],
        'humidity': weather_data['hourly']['relativehumidity_2m'],
        'windspeed': weather_data['hourly']['windspeed_10m'],
        'precipitation': weather_data['hourly']['precipitation']
    })
    
    # Sort by timestamp
    weather_df = weather_df.sort_values('timestamp').reset_index(drop=True)

    return weather_df



def preprocess_pipeline():
    """
    Master function that:
    1. Cleans all raw data
    2. Merges everything together
    3. Saves files
    """
    
    # Clean individual datasets 
    rt_lmp = clean_RT_data(glob.glob(str(base / '*RT_prices*.csv')))
    rt_lmp.to_parquet(interim / 'rt_cleaned.parquet')
    del rt_lmp

    dam_lmp = clean_DAM_data(glob.glob(str(base / '*DAM_prices*.csv')))
    dam_lmp.to_parquet(interim / 'DAM_cleaned.parquet')     
    del dam_lmp

    prc = clean_prc(pre_rtcb_dir= base / 'ercot_archive/', post_rtcb_filepath= base / 'ercot_archive/post_prc.csv')
    prc.to_parquet(interim / 'prc_cleaned.parquet')
    del prc

    load = clean_load_data(glob.glob(str(base / '*load_data*.csv')))
    load.to_parquet(interim / 'load_cleaned.parquet') 
    del load

    weather = fetch_weather_data()
    weather.to_parquet(interim / 'weather_cleaned.parquet')   
    del weather

    outages = clean_outages_data(glob.glob(str(base / '*outages*.csv')))
    outages.to_parquet(interim / 'outages_cleaned.parquet') 
    del outages

    solar = clean_solar(glob.glob(str(base / '*solar*.csv')))
    solar.to_parquet(interim / 'solar_cleaned.parquet') 
    del solar

    wind = clean_wind_data(glob.glob(str(base / '*wind*.csv')))
    wind.to_parquet(interim / 'wind_cleaned.parquet') 
    del wind

    fcst = clean_fcst_data(glob.glob(str(base / '*load_fcst_data*.csv')))
    fcst.to_parquet(interim / 'fcst_cleaned.parquet')
    del fcst
    
    
    # STEP 5: Merge everything
    date_range = pd.date_range(start='2021-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h')
    merged = pd.DataFrame({'timestamp': date_range})
    hour_ending_files = ['rt_cleaned', 'DAM_cleaned', 'load_cleaned', 'outages_cleaned', 'solar_cleaned', 'wind_cleaned', 'fcst_cleaned']

    for f in ['rt_cleaned', 'DAM_cleaned', 'load_cleaned', 'weather_cleaned', 'outages_cleaned', 'solar_cleaned', 'wind_cleaned', 'fcst_cleaned', 'prc_cleaned']:
        df = pd.read_parquet(interim / (f + '.parquet'))
        if f in hour_ending_files:
            df['timestamp'] = df['timestamp'] - pd.Timedelta(hours=1)
        merged = merged.merge(df, on='timestamp', how='left')
        del df

    merged['RT_DAM_spread'] = merged['RT_price'] - merged['DAM_price']
    merged.to_parquet(interim / 'merged_all_data.parquet')
    
    return merged

if __name__ == '__main__':
    preprocess_pipeline()
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

    # Append GridStatus.io API parquets (post-RTC+B going-forward pulls, already hourly)
    gs_dir = BASE_DIR / 'data' / 'raw' / 'gridstatus'
    for f in sorted(gs_dir.glob('rt_prices_*.parquet')):
        gs = pd.read_parquet(f)
        if 'interval_start_utc' in gs.columns:
            gs['timestamp'] = (
                pd.to_datetime(gs['interval_start_utc'], utc=True)
                .dt.tz_convert('America/Chicago')
                .dt.tz_localize(None)
            )
            gs = gs.rename(columns={'spp': 'RT_price'})[['timestamp', 'RT_price']]
            df_rt_hourly = pd.concat([df_rt_hourly, gs], ignore_index=True)
            print(f"  + GridStatus RT: {len(gs)} rows from {f.name}")

    df_rt_hourly = df_rt_hourly.drop_duplicates(subset='timestamp', keep='first')
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

    # Merge gap-fill data pulled from gridstatus (covers days missing from scraped CSVs)
    # Gap fill uses hour-beginning (Interval Start), but CSV data uses hour-ending
    # (from fix_hour24_helper). Add 1h so the merge step's -1h adjustment treats both
    # sources consistently.
    gap_fill_path = base / 'dam_gap_fill.parquet'
    if gap_fill_path.exists():
        gf = pd.read_parquet(gap_fill_path)
        gf = gf.rename(columns={'Interval Start': 'timestamp', 'SPP': 'DAM_price'})
        gf = gf[['timestamp', 'DAM_price']].copy()
        gf['timestamp'] = pd.to_datetime(gf['timestamp']) + pd.Timedelta(hours=1)
        df_da_clean = pd.concat([df_da_clean, gf], ignore_index=True)

    df_da_clean = df_da_clean.drop_duplicates(subset='timestamp', keep='first')

    # Append GridStatus.io API parquets (post-RTC+B going-forward pulls, already hourly)
    # GridStatus uses hour-beginning (interval_start_utc). CSV data uses hour-ending.
    # Add +1h so the merge step's -1h adjustment produces correct hour-beginning alignment
    # (same convention as dam_gap_fill).
    gs_dir = BASE_DIR / 'data' / 'raw' / 'gridstatus'
    for f in sorted(gs_dir.glob('dam_prices_*.parquet')):
        gs = pd.read_parquet(f)
        if 'interval_start_utc' in gs.columns:
            gs['timestamp'] = (
                pd.to_datetime(gs['interval_start_utc'], utc=True)
                .dt.tz_convert('America/Chicago')
                .dt.tz_localize(None)
                + pd.Timedelta(hours=1)
            )
            gs = gs.rename(columns={'spp': 'DAM_price'})[['timestamp', 'DAM_price']]
            df_da_clean = pd.concat([df_da_clean, gs], ignore_index=True)
            print(f"  + GridStatus DAM: {len(gs)} rows from {f.name}")

    df_da_clean = df_da_clean.drop_duplicates(subset='timestamp', keep='first')

    return df_da_clean

def clean_ordc_adders(pre_rtcb_dir, post_rtcb_dir):
    """
    Build unified ORDC/RDPA price adder series from pre and post RTC+B sources.

    Pre-RTC+B (2021 – Dec 4, 2025): ERCOT archive xlsx, 5-min SCED intervals.
        RTORPA  – ORDC Reserve Price Adder ($/MWh), fired when PRC is low
        RTOFFPA – Off-line reserve adder ($/MWh)
        RTORDPA – ORDC Deployment Price Adder ($/MWh)

    Post-RTC+B (Dec 5, 2025+): GridStatus.io API parquet files.
        rtrdpa  – Reliability Deployment Price Adder, functional analog to RTORPA
        RTOFFPA / RTORDPA have no clean post-RTC+B equivalent → NaN

    Both sources aggregated from 5-min to hourly mean.
    Output columns: ['timestamp', 'RTORPA', 'RTOFFPA', 'RTORDPA']
    """
    ADDER_COLS = ['SCED Timestamp', 'RTORPA', 'RTOFFPA', 'RTORDPA']

    # 1. Pre-RTC+B: read xlsx archive files
    all_years = []
    for filepath in sorted(Path(pre_rtcb_dir).glob("price_adders_*.xlsx")):
        sheet_dict = pd.read_excel(filepath, sheet_name=None)
        all_sheets = []
        for name, sheet in sheet_dict.items():
            if name.lower() in ['report info', 'sheet1']:
                continue
            sheet.columns = sheet.iloc[7].values
            sheet = sheet.iloc[8:].reset_index(drop=True)
            sheet.columns = sheet.columns.str.strip()
            sheet = sheet[ADDER_COLS].copy()
            sheet['SCED Timestamp'] = pd.to_datetime(sheet['SCED Timestamp'])
            sheet['hour'] = sheet['SCED Timestamp'].dt.floor('h')
            sheet = sheet.groupby('hour').agg({'RTORPA': 'mean', 'RTOFFPA': 'mean', 'RTORDPA': 'mean'}).reset_index()
            all_sheets.append(sheet)
        all_years.append(pd.concat(all_sheets).reset_index(drop=True))

    pre = pd.concat(all_years).reset_index(drop=True)
    for col in ['RTORPA', 'RTOFFPA', 'RTORDPA']:
        pre[col] = pd.to_numeric(pre[col], errors='coerce')
    pre = pre.rename(columns={'hour': 'timestamp'})

    # 2. Post-RTC+B: GridStatus.io parquet files
    post_files = sorted(Path(post_rtcb_dir).glob("price_adders_*.parquet"))
    if post_files:
        post_dfs = []
        for f in post_files:
            df = pd.read_parquet(f)
            # Normalize column names — GridStatus.io may return ERCOT-native casing
            # (SCEDTimestamp, RTRDPA) or lowercase (sced_timestamp_utc, rtrdpa)
            col_map = {c: c.lower() for c in df.columns}
            df = df.rename(columns=col_map)
            # Determine timestamp column and parse
            if 'sced_timestamp_utc' in df.columns:
                df['timestamp'] = (
                    pd.to_datetime(df['sced_timestamp_utc'], utc=True)
                    .dt.tz_convert('America/Chicago')
                    .dt.tz_localize(None)
                    .dt.floor('h')
                )
            elif 'scedtimestamp' in df.columns:
                # Market-local naive datetime string
                df['timestamp'] = pd.to_datetime(df['scedtimestamp']).dt.floor('h')
            else:
                raise KeyError(f"No recognized timestamp column in {f.name}: {list(df.columns)}")
            # rtrdpa is the post-RTC+B analog to RTORPA; RTOFFPA/RTORDPA have no equivalent
            df['rtrdpa'] = pd.to_numeric(df['rtrdpa'], errors='coerce')
            df = df.groupby('timestamp').agg(RTORPA=('rtrdpa', 'mean')).reset_index()
            df['RTOFFPA'] = np.nan
            df['RTORDPA'] = np.nan
            post_dfs.append(df)
        post = pd.concat(post_dfs).reset_index(drop=True)
    else:
        post = pd.DataFrame(columns=['timestamp', 'RTORPA', 'RTOFFPA', 'RTORDPA'])

    # 3. Concat pre + post, sort, deduplicate
    adders = pd.concat([pre, post]).sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
    return adders[['timestamp', 'RTORPA', 'RTOFFPA', 'RTORDPA']]


def clean_prc(pre_rtcb_dir, post_rtcb_dir):
    """
    Build unified PRC series from pre and post RTC+B sources.
    Pre: ERCOT archive xlsx files (2021 - Dec 4, 2025)
    Post: legacy post_prc.csv in ercot_archive/ (manually downloaded) plus
          any prc_*.parquet files saved by pull_prc() in data/raw/gridstatus/
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

    # 2. Read post-RTC+B — legacy CSV first, then any API-pulled parquets on top
    post_dfs = []
    legacy_csv = Path(pre_rtcb_dir) / 'post_prc.csv'
    if legacy_csv.exists():
        df = pd.read_csv(legacy_csv)
        df['timestamp'] = pd.to_datetime(df['interval_start_local']).dt.tz_localize(None)
        post_dfs.append(df[['timestamp', 'prc']].rename(columns={'prc': 'PRC'}))

    for f in sorted(Path(post_rtcb_dir).glob("prc_*.parquet")):
        df = pd.read_parquet(f)
        # Handle three schema versions from GridStatus.io:
        # 1. 'interval_start_local' — single pull with timezone='market'
        # 2. 'time_local' — batch pull via gridstatusio client
        # 3. 'interval_start_utc' — hourly resampled pull (preferred)
        if 'interval_start_utc' in df.columns:
            df['timestamp'] = (
                pd.to_datetime(df['interval_start_utc'], utc=True)
                .dt.tz_convert('America/Chicago')
                .dt.tz_localize(None)
            )
        elif 'interval_start_local' in df.columns:
            df['timestamp'] = pd.to_datetime(df['interval_start_local']).dt.tz_localize(None)
        elif 'time_local' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_local']).dt.tz_localize(None)
        else:
            raise KeyError(f"No recognized timestamp column in {f.name}: {list(df.columns)}")
        post_dfs.append(df[['timestamp', 'prc']].rename(columns={'prc': 'PRC'}))

    # 3. Aggregate post-RTC+B to hourly min (same as pre-RTC+B)
    # PRC data comes in at 5-min SCED intervals. Hourly min = most stressed moment
    # in the hour (conservative for curtailment decisions).
    if post_dfs:
        post = pd.concat(post_dfs).sort_values('timestamp').reset_index(drop=True)
        post['PRC'] = pd.to_numeric(post['PRC'], errors='coerce')
        post['hour'] = post['timestamp'].dt.floor('h')
        post = post.groupby('hour').agg({'PRC': 'min'}).reset_index()
        post = post.rename(columns={'hour': 'timestamp'})
        prc = pd.concat([pre, post]).sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
    else:
        prc = pre.sort_values('timestamp').reset_index(drop=True)

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

    # Append GridStatus.io API parquets (zone-level load, already hourly)
    # Uses hour-beginning; add +1h for hour-ending convention (merge step subtracts 1h).
    gs_dir = BASE_DIR / 'data' / 'raw' / 'gridstatus'
    for f in sorted(gs_dir.glob('load_zone_*.parquet')):
        gs = pd.read_parquet(f)
        if 'interval_start_utc' in gs.columns and 'west' in gs.columns:
            gs['timestamp'] = (
                pd.to_datetime(gs['interval_start_utc'], utc=True)
                .dt.tz_convert('America/Chicago')
                .dt.tz_localize(None)
                + pd.Timedelta(hours=1)
            )
            gs = gs.rename(columns={'west': 'hub_load', 'system_total': 'load_total'})
            gs = gs[['timestamp', 'hub_load', 'load_total']]
            df_load_clean = pd.concat([df_load_clean, gs], ignore_index=True)
            print(f"  + GridStatus load: {len(gs)} rows from {f.name}")

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
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
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

def fetch_weather_data(start_date='2021-01-01', end_date=None,
                       lat=31.99, lon=-102.08, location_name='Midland, TX'):
    """
    Fetch weather data from Open-Meteo using two sources:
    1. Archive API (ERA5 reanalysis) — historical bulk, ~5 day lag
    2. Forecast API — recent days through present, bridges the ERA5 gap

    Archive data is preferred where both overlap (reanalysis is more accurate).
    Uses Midland, Texas coordinates (center of West Texas HUB).

    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date for archive pull (YYYY-MM-DD). Defaults to 5 days ago.
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

    # 1. Archive API — historical bulk up to ~5 days ago
    if end_date is None:
        end_date = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    archive_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation"],
        "timezone": "America/Chicago"
    }

    response = requests.get(archive_url, params=archive_params)
    if response.status_code != 200:
        print(f"  Archive API failed with status code {response.status_code}")
        return None

    archive_data = response.json()
    archive_df = pd.DataFrame({
        'timestamp': pd.to_datetime(archive_data['hourly']['time']),
        'temperature': archive_data['hourly']['temperature_2m'],
        'humidity': archive_data['hourly']['relativehumidity_2m'],
        'windspeed': archive_data['hourly']['windspeed_10m'],
        'precipitation': archive_data['hourly']['precipitation']
    })

    # 2. Forecast API — recent days through present to bridge ERA5 lag
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"],
        "timezone": "America/Chicago",
        "past_days": 10,
        "forecast_days": 2
    }

    forecast_response = requests.get(forecast_url, params=forecast_params)
    if forecast_response.status_code == 200:
        forecast_data = forecast_response.json()
        forecast_df = pd.DataFrame({
            'timestamp': pd.to_datetime(forecast_data['hourly']['time']),
            'temperature': forecast_data['hourly']['temperature_2m'],
            'humidity': forecast_data['hourly']['relative_humidity_2m'],
            'windspeed': forecast_data['hourly']['wind_speed_10m'],
            'precipitation': forecast_data['hourly']['precipitation']
        })
        # Concat, preferring archive (reanalysis) where both overlap
        weather_df = pd.concat([archive_df, forecast_df]).drop_duplicates(
            subset='timestamp', keep='first'
        )
    else:
        print(f"  Forecast API failed ({forecast_response.status_code}), using archive only")
        weather_df = archive_df

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

    prc = clean_prc(pre_rtcb_dir= base / 'ercot_archive/', post_rtcb_dir= BASE_DIR / 'data' / 'raw' / 'gridstatus')
    prc.to_parquet(interim / 'prc_cleaned.parquet')
    del prc

    ordc = clean_ordc_adders(pre_rtcb_dir= base / 'ercot_archive/', post_rtcb_dir= BASE_DIR / 'data' / 'raw' / 'gridstatus')
    ordc.to_parquet(interim / 'ordc_adders_cleaned.parquet')
    del ordc

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

    fcst = clean_fcst_data(glob.glob(str(base / '*load_fcst_data*.csv')) + glob.glob(str(base / '*load_fcst_data*.parquet')))
    fcst.to_parquet(interim / 'fcst_cleaned.parquet')
    del fcst
    
    # DAM system-wide price (HB_HUBAVG) for congestion spread calculation.
    # Pulled from gridstatus get_dam_spp() — same archive as HB_WEST DAM.
    # Timestamps are hour-beginning; add +1h so the merge step's -1h adjustment
    # produces correct hour-beginning alignment (same convention as dam_gap_fill).
    dam_sys_path = BASE_DIR / 'data' / 'raw' / 'gridstatus' / 'dam_hubavg_2021_2025.parquet'
    if dam_sys_path.exists():
        dam_sys = pd.read_parquet(dam_sys_path)
        dam_sys['timestamp'] = pd.to_datetime(dam_sys['timestamp']) + pd.Timedelta(hours=1)
        dam_sys.to_parquet(interim / 'dam_system_cleaned.parquet')
        del dam_sys
        print("  DAM system lambda cleaned ✓")
    else:
        print("  ⚠ DAM system file not found — run pull_dam_hubavg() first")

    # Waha Hub natural gas spot price (daily, from EIA API).
    # Gas price sets the marginal cost of electricity when wind drops — a key driver of
    # price spikes, especially for the 24h model where real-time signals aren't available.
    # Daily price forward-filled to hourly (doesn't change intra-day).
    gas_path = BASE_DIR / 'data' / 'raw' / 'gridstatus' / 'gas_price_waha_2021_2025.parquet'
    if gas_path.exists():
        gas = pd.read_parquet(gas_path)
        gas['timestamp'] = pd.to_datetime(gas['date'])
        gas = gas[['timestamp', 'gas_price_waha']].sort_values('timestamp')
        gas.to_parquet(interim / 'gas_price_cleaned.parquet')
        del gas
        print("  Gas price (Waha Hub) cleaned ✓")
    else:
        print("  ⚠ Gas price file not found — run EIA pull first")

    # STEP 5: Merge everything
    # End date derived from the latest timestamp in the cleaned data so new pulls are included automatically
    _end = max(
        pd.read_parquet(interim / f).index.max() if pd.read_parquet(interim / f).index.name == 'timestamp'
        else pd.read_parquet(interim / f)['timestamp'].max()
        for f in ['rt_cleaned.parquet', 'prc_cleaned.parquet', 'load_cleaned.parquet']
    )
    date_range = pd.date_range(start='2021-01-01 00:00:00', end=_end, freq='h')
    merged = pd.DataFrame({'timestamp': date_range})
    hour_ending_files = ['DAM_cleaned', 'load_cleaned', 'outages_cleaned', 'solar_cleaned', 'wind_cleaned', 'fcst_cleaned']

    merge_files = ['rt_cleaned', 'DAM_cleaned', 'load_cleaned', 'weather_cleaned', 'outages_cleaned', 'solar_cleaned', 'wind_cleaned', 'fcst_cleaned', 'prc_cleaned', 'ordc_adders_cleaned']
    if (interim / 'dam_system_cleaned.parquet').exists():
        merge_files.append('dam_system_cleaned')
        hour_ending_files.append('dam_system_cleaned')
    for f in merge_files:
        df = pd.read_parquet(interim / (f + '.parquet'))
        if f in hour_ending_files:
            df['timestamp'] = df['timestamp'] - pd.Timedelta(hours=1)
        merged = merged.merge(df, on='timestamp', how='left')
        del df

    merged['RT_DAM_spread'] = merged['RT_price'] - merged['DAM_price']

    # Gas price: daily → hourly via merge_asof (each hour gets the most recent daily price)
    if (interim / 'gas_price_cleaned.parquet').exists():
        gas = pd.read_parquet(interim / 'gas_price_cleaned.parquet')
        gas['timestamp'] = pd.to_datetime(gas['timestamp'])
        merged = pd.merge_asof(
            merged.sort_values('timestamp'),
            gas.sort_values('timestamp'),
            on='timestamp',
            direction='backward'  # each hour gets the most recent daily price
        )
        print(f"  Gas price merged: {merged['gas_price_waha'].notna().sum()} / {len(merged)} non-null")

    merged.to_parquet(interim / 'merged_all_data.parquet')
    
    return merged

if __name__ == '__main__':
    preprocess_pipeline()
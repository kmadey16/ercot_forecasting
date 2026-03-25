import pandas as pd
from gridstatus import Ercot
from pathlib import Path

#Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'gridstatus'

settlement_point = 'HB_WEST'

# Pull functions (new data)

def pull_prc(start, end):
    
    """Post-RTC+B PRC only (Dec 5, 2025+)""" # FROM GRIDSTATUSIO System-wide
    
    from gridstatusio import GridStatusClient
    import os
    
    client = GridStatusClient(api_key=os.environ.get("GRIDSTATUS_API_KEY"))

    df = client.get_dataset(
    dataset="ercot_prc",
    start=start,
    end=end,
    timezone="market",
    limit=100000)

    df.to_parquet(RAW_DIR / f"prc_{start}_{end}.parquet", index=False)
    print(f"PRC: {len(df)} rows")

def pull_price_adders(start,end): #post RTC+B price adders using GridstatusIO
    from gridstatusio import GridStatusClient
    import os

    client = GridStatusClient(api_key=os.environ.get("GRIDSTATUS_API_KEY"))

    df = client.get_dataset(
    dataset="ercot_real_time_adders",
    start=start,
    end=end,
    timezone="market",
    limit=100000)

    df.to_parquet(RAW_DIR / f"price_adders_{start}_{end}.parquet", index=False)
    print(f"price_adders: {len(df)} rows")



def pull_load(ercot, start, end):
    
    """System-wide actual load (hourly)."""
    df = ercot.get_load(date=start, end=end, verbose=True)
    
    df.to_parquet(RAW_DIR / f"load_{start}_{end}.parquet", index=False)
    print(f"Load: {len(df)} rows")

def pull_load_forecast(ercot, start, end):
    """Load forecast"""
    
    df = ercot.get_load_forecast(date=start, end=end, verbose=True)
    
    df.to_parquet(RAW_DIR / f"load_forecast_{start}_{end}.parquet", index=False)
    print(f"Load_forecast: {len(df)} rows")

def pull_wind(ercot, start, end):
    """Wind actual + forecast"""

    df = ercot.get_hourly_wind_report(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"wind_{start}_{end}.parquet", index=False)
    print(f"Wind: {len(df)} rows")

def pull_solar(ercot, start, end):
    """Solar actual + forecast"""
    
    df = ercot.get_hourly_solar_report(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"solar_{start}_{end}.parquet", index=False)
    print(f"Solar: {len(df)} rows")

def pull_outages(ercot, start, end):
    """Hourly resource outage capacity"""

    df = ercot.get_hourly_resource_outage_capacity(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"outages_{start}_{end}.parquet", index=False)
    print(f"Outages: {len(df)} rows")

def pull_rt_prices(ercot, start, end, location=settlement_point):
    """RT settlement point prices for backtesting."""
    df = ercot.get_spp(date=start, end=end, market="REAL_TIME_15_MIN", locations=[location], verbose=True)
    df.to_parquet(RAW_DIR / f"rt_prices_{start}_{end}.parquet", index=False)
    print(f"RT prices: {len(df)} rows")

def pull_dam_prices(ercot, start, end, location=settlement_point):
    """DA settlement point prices for backtesting."""
    df = ercot.get_spp(date=start, end=end, market="DAY_AHEAD_HOURLY", locations=[location], verbose=True)
    df.to_parquet(RAW_DIR / f"dam_prices_{start}_{end}.parquet", index=False)
    print(f"DAM prices: {len(df)} rows")

def pull_as_prices(ercot, start, end):
    """DAM ancillary service clearing prices."""
    df = ercot.get_as_prices(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"as_prices_{start}_{end}.parquet", index=False)
    print(f"AS prices: {len(df)} rows")

def pull_dam_system_lambda(ercot, start, end):
    """Day-ahead system lambda."""
    df = ercot.get_dam_system_lambda(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"dam_system_lambda_{start}_{end}.parquet", index=False)
    print(f"DAM system lambda: {len(df)} rows")

def pull_temperature(ercot, start, end):
    """Temperature forecast by weather zone."""
    df = ercot.get_temperature_forecast_by_weather_zone(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"temperature_{start}_{end}.parquet", index=False)
    print(f"Temperature: {len(df)} rows")

def pull_system_lambda(ercot, start, end):
    """RT SCED system lambda (5-min)."""
    df = ercot.get_sced_system_lambda(date=start, end=end, verbose=True)
    df.to_parquet(RAW_DIR / f"system_lambda_{start}_{end}.parquet", index=False)
    print(f"System lambda: {len(df)} rows")


# Orchestrators

def backfill_prices(ercot, years, location=settlement_point):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in years:
        # RT prices
        rt_prices = ercot.get_rtm_spp(year,verbose=True)
        rt_prices = rt_prices[rt_prices['Location'] == location].reset_index(drop=True)

        #DAM prices
        dam_prices = ercot.get_dam_spp(year,verbose=True)
        dam_prices = dam_prices[dam_prices['Location'] == location].reset_index(drop=True)
        
        #Output to raw
        rt_prices.to_parquet(RAW_DIR / f"RT_prices_{year}.parquet", index=False)
        
        dam_prices.to_parquet(RAW_DIR / f"DAM_prices_{year}.parquet", index=False)

        print(f"{year}: RT={len(rt_prices)} rows, DAM={len(dam_prices)} rows")


def pull_new_data(start, end):
    
    """Going-forward pull for all datasets."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ercot = Ercot()

    pull_load(ercot, start, end)
    pull_load_forecast(ercot, start, end)
    pull_wind(ercot, start, end)
    pull_solar(ercot, start, end)
    pull_outages(ercot, start, end)
    pull_temperature(ercot, start, end)
    pull_system_lambda(ercot, start, end)
    pull_dam_system_lambda(ercot, start, end)
    pull_as_prices(ercot, start, end)
    pull_rt_prices(ercot, start, end)
    pull_dam_prices(ercot, start, end)
    pull_prc(start, end)  # gridstatusio
    #pull_price_adders(start,end) #gridstatusio (dont run yet)

    print("Done")
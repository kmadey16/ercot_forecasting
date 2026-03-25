
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_models():
    lr_1h = joblib.load(PROJECT_ROOT / 'models/lr_1h_prc.pkl')
    lgbm_1h = joblib.load(PROJECT_ROOT / 'models/lgbm_1h_residual.pkl')
    lr_24h = joblib.load(PROJECT_ROOT / 'models/lr_24h_prc.pkl')
    return lr_1h, lgbm_1h, lr_24h

def load_thresholds():
    with open(PROJECT_ROOT / 'models/thresholds.json', 'r') as f:
        return json.load(f)

def predict_prc(df, lr_model, lgbm_model=None, drop_cols=None):
    X = df.drop(columns=drop_cols)
    
    preds = lr_model.predict(X)
    if lgbm_model is not None:
        preds += lgbm_model.predict(X)
    
    df['predicted_prc'] = preds
    
    bins = [0, 3000, 5000, 10000, float('inf')]
    labels = ['Scarcity', 'Tight', 'Normal', 'Surplus']
    df['predicted_regime'] = pd.cut(preds, bins=bins, labels=labels)
    
    return df

def dispatch_action(predicted_prc, tight_threshold):
    if predicted_prc < 3000:
        return 'CURTAIL — emergency, shut down all flexible load'
    elif predicted_prc < tight_threshold:
        return 'REDUCE — lower non-critical load'
    else:
        return 'OPERATE — standard operations'

if __name__ == '__main__':
    lr_1h, lgbm_1h, lr_24h = load_models()
    thresholds = load_thresholds()
    
    df = pd.read_parquet(PROJECT_ROOT / 'data/processed/model_ready.parquet')
    
    drop_1h = ['timestamp', 'PRC', 'regime', 'RT_price', 'DAM_price', 'RT_DAM_spread']
    
    drop_24h = [
        'timestamp', 'PRC', 'regime', 'RT_price', 'DAM_price', 'RT_DAM_spread',
        'hub_load', 'load_total',
        'WGRPP_LZ_WEST', 'WGRPP_SYSTEM_WIDE', 'PVGRPP_SYSTEM_WIDE',
        'net_load_system', 'net_load_west',
        'renewable_pct_system', 'renewable_pct_west',
        'wind_forecast_error', 'solar_forecast_error',
        'RT_1h_lag', 'DAM_1h_lag', 'PRC_1h_lag',
        'RT_price_roll_mean_6h', 'RT_price_roll_std_6h',
        'DAM_price_roll_mean_6h', 'DAM_price_roll_std_6h',
        'hub_load_roll_mean_6h', 'hub_load_roll_std_6h',
        'load_total_roll_mean_6h', 'load_total_roll_std_6h',
        'WGRPP_LZ_WEST_roll_mean_6h', 'WGRPP_LZ_WEST_roll_std_6h',
        'load_total_ramp', 'hub_load_ramp',
        'wind_west_ramp', 'wind_system_ramp',
        'RT_price_ramp'
    ]
    
    # 1h predictions (ensemble)
    t_1h = thresholds['mining_1h']['tight_threshold']
    results_1h = predict_prc(df.copy(), lr_1h, lgbm_1h, drop_1h)
    results_1h['action'] = results_1h['predicted_prc'].apply(lambda x: dispatch_action(x, t_1h))
    print('--- 1h Model ---')
    print(results_1h[['timestamp', 'predicted_prc', 'predicted_regime', 'action']].tail(24))
    
    # 24h predictions (LR only)
    t_24h = thresholds['mining_24h']['tight_threshold']
    results_24h = predict_prc(df.copy(), lr_24h, lgbm_model=None, drop_cols=drop_24h)
    results_24h['action'] = results_24h['predicted_prc'].apply(lambda x: dispatch_action(x, t_24h))
    print('\n--- 24h Model ---')
    print(results_24h[['timestamp', 'predicted_prc', 'predicted_regime', 'action']].tail(24))
import joblib
import pandas as pd
import os

PROJECT_ROOT = '/Users/kamilmadey/Desktop/ercot_forecasting_project/'

def load_models():
    model_1h = joblib.load(PROJECT_ROOT + 'models/lgbm_1h_regime.pkl')
    model_24h = joblib.load(PROJECT_ROOT + 'models/lgbm_24h_regime.pkl')
    return model_1h, model_24h

def predict_regime(df, model, drop_cols):
    X = df.drop(columns=drop_cols)
    df['predicted_regime'] = model.predict(X)
    return df

def dispatch_action(regime):
    actions = {
        'Low': 'FULL POWER — cheap electricity, maximize consumption',
        'Normal': 'OPERATE — standard operations',
        'Tight': 'REDUCE — lower non-critical load',
        'Scarcity': 'CURTAIL — shut down flexible load'
    }
    return actions[regime]

if __name__ == '__main__':
    
    model_1h, model_24h = load_models()
    
    df = pd.read_parquet(PROJECT_ROOT + 'data/processed/model_ready.parquet')
    
    drop_1h = ['timestamp', 'RT_price', 'RT_DAM_spread', 'RT_price_ramp', 'regime']
    
    drop_24h = drop_1h + [
        'RT_1h_lag', 'DAM_1h_lag',
        'RT_price_roll_mean_6h', 'RT_price_roll_std_6h',
        'DAM_price_roll_mean_6h', 'DAM_price_roll_std_6h',
        'hub_load_roll_mean_6h', 'hub_load_roll_std_6h',
        'load_total_roll_mean_6h', 'load_total_roll_std_6h',
        'WGRPP_LZ_WEST_roll_mean_6h', 'WGRPP_LZ_WEST_roll_std_6h',
        'load_total_ramp', 'hub_load_ramp',
        'wind_west_ramp', 'wind_system_ramp',
        'hub_load', 'load_total',
        'WGRPP_LZ_WEST', 'WGRPP_SYSTEM_WIDE',
        'PVGRPP_SYSTEM_WIDE',
        'net_load_system', 'net_load_west',
        'renewable_pct_system', 'renewable_pct_west'
    ]
    
    results_1h = predict_regime(df.copy(), model_1h, drop_1h)
    results_1h['action'] = results_1h['predicted_regime'].map(dispatch_action)
    print('--- 1h Model ---')
    print(results_1h[['timestamp', 'predicted_regime', 'action']].tail(24))
    
    results_24h = predict_regime(df.copy(), model_24h, drop_24h)
    results_24h['action'] = results_24h['predicted_regime'].map(dispatch_action)
    print('\n--- 24h Model ---')
    print(results_24h[['timestamp', 'predicted_regime', 'action']].tail(24))
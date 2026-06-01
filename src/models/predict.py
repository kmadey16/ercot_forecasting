
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_models():
    """Load all production models (PRC + price + spread + 4CP)."""
    lr_1h = joblib.load(PROJECT_ROOT / 'models/lr_1h_prc_v2.pkl')
    lgbm_1h = joblib.load(PROJECT_ROOT / 'models/lgbm_1h_residual_v2.pkl')
    lr_24h = joblib.load(PROJECT_ROOT / 'models/lr_24h_prc_v2.pkl')
    price_1h = joblib.load(PROJECT_ROOT / 'models/lgbm_price_1h_v1.pkl')
    spread_24h = joblib.load(PROJECT_ROOT / 'models/lgbm_spread_24h_v1.pkl')
    cp4 = joblib.load(PROJECT_ROOT / 'models/lgbm_4cp_v1.pkl')
    return lr_1h, lgbm_1h, lr_24h, price_1h, spread_24h, cp4


def load_thresholds():
    with open(PROJECT_ROOT / 'models/thresholds.json', 'r') as f:
        return json.load(f)


def load_price_metadata():
    with open(PROJECT_ROOT / 'models/lgbm_price_1h_v1_metadata.json', 'r') as f:
        return json.load(f)


def load_spread_metadata():
    with open(PROJECT_ROOT / 'models/lgbm_spread_24h_v1_metadata.json', 'r') as f:
        return json.load(f)


def load_4cp_metadata():
    with open(PROJECT_ROOT / 'models/lgbm_4cp_v1_metadata.json', 'r') as f:
        return json.load(f)


def predict_prc(df, lr_model, lgbm_model=None, feature_cols=None, drop_cols=None):
    """Predict PRC using LR (+ optional LGBM residual).

    Uses feature_cols from model metadata if provided (ensures exact match with
    training features). Falls back to drop_cols for backwards compatibility.
    """
    if feature_cols is not None:
        X = df[feature_cols]
    else:
        X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

    preds = lr_model.predict(X)
    if lgbm_model is not None:
        preds += lgbm_model.predict(X)

    df['predicted_prc'] = preds

    bins = [0, 3000, 5000, 10000, float('inf')]
    labels = ['Scarcity', 'Tight', 'Normal', 'Surplus']
    df['predicted_regime'] = pd.cut(preds, bins=bins, labels=labels)

    return df


def predict_price(df, price_model, feature_cols, price_offset):
    """Predict RT price using the Layer 2 price model.

    Returns predicted price in raw $/MWh (reverse-transformed from log1p space).
    """
    X = df[feature_cols].dropna()
    pred_log = price_model.predict(X)
    pred_raw = np.expm1(pred_log) - price_offset

    df.loc[X.index, 'predicted_price'] = pred_raw
    return df


def predict_spread(df, spread_model, feature_cols):
    """Predict RT-DAM spread using the 24h spread model.

    Returns predicted spread in raw $/MWh and reconstructed RT price (DAM + spread).
    Advisory signal — informs planning, does not trigger curtailment alone.
    """
    X = df[feature_cols]
    pred_spread = spread_model.predict(X)

    df['predicted_spread'] = pred_spread
    df['predicted_rt_24h'] = df['DAM_price'] + pred_spread
    return df


def advisory_24h(predicted_prc, predicted_rt_24h, dam_price, tight_threshold):
    """24h planning advisory — informs scheduling, does NOT trigger curtailment.

    Combines 24h PRC forecast + DAM price + spread prediction into a risk assessment.
    Operators use this to pre-position (migrate workloads, preserve battery SoC)
    but actual curtailment decisions happen at the 1h layer.
    """
    risks = []

    if predicted_prc < 3000:
        risks.append('PRC scarcity expected')
    elif predicted_prc < tight_threshold:
        risks.append('PRC tight expected')

    if dam_price is not None and dam_price > 100:
        risks.append(f'DAM high (${dam_price:.0f})')

    if predicted_rt_24h is not None and predicted_rt_24h > 100:
        risks.append(f'RT may exceed ${predicted_rt_24h:.0f}')

    if risks:
        return f'ELEVATED RISK — {", ".join(risks)}. Pre-position flexible load.'
    return 'LOW RISK — standard scheduling'


def predict_4cp(df, cp4_model, feature_cols):
    """Predict 4CP peak probability.

    Only meaningful for June-September, 2-7 PM. Outside this window, probability
    is set to 0. Returns P(this hour is the monthly coincident peak).
    """
    proba = np.zeros(len(df))

    # Only score summer afternoon hours
    summer_mask = (df['month'].isin([6, 7, 8, 9])) & (df['hour'].between(14, 19))
    if summer_mask.sum() > 0:
        X = df.loc[summer_mask, feature_cols]
        proba[summer_mask.values] = cp4_model.predict_proba(X)[:, 1]

    df['p_4cp'] = proba
    return df


def dispatch_action(predicted_prc, predicted_price, tight_threshold,
                    price_threshold=None):
    """Combined Layer 1 + Layer 2 dispatch decision.

    Checks both PRC (system-wide reserves) and predicted price (local congestion).
    Returns the MORE AGGRESSIVE action of the two signals:
      - PRC scarcity (<3000 MW) always wins → CURTAIL (emergency)
      - PRC tight OR price spike → REDUCE
      - Neither triggered → OPERATE

    The price threshold acts as an independent trigger: even if PRC says "Normal",
    a predicted price above the threshold means local congestion is likely and
    you should reduce load to avoid paying elevated rates.
    """
    # Layer 1: PRC signal
    if predicted_prc < 3000:
        return 'CURTAIL — emergency, shut down all flexible load'

    prc_says_reduce = predicted_prc < tight_threshold

    # Layer 2: Price signal (if threshold is set)
    price_says_reduce = False
    if price_threshold is not None and predicted_price is not None:
        price_says_reduce = predicted_price > price_threshold

    if prc_says_reduce or price_says_reduce:
        if prc_says_reduce and price_says_reduce:
            return 'REDUCE — PRC tight + price spike, lower non-critical load'
        elif prc_says_reduce:
            return 'REDUCE — PRC tight, lower non-critical load'
        else:
            return 'REDUCE — price spike, lower non-critical load'

    return 'OPERATE — standard operations'


if __name__ == '__main__':
    lr_1h, lgbm_1h, lr_24h, price_1h, spread_24h, cp4 = load_models()
    thresholds = load_thresholds()
    price_meta = load_price_metadata()
    spread_meta = load_spread_metadata()
    cp4_meta = load_4cp_metadata()

    # Load PRC model feature lists from metadata
    with open(PROJECT_ROOT / 'models/lr_1h_prc_v2_metadata.json') as f:
        prc_1h_features = json.load(f)['features']
    with open(PROJECT_ROOT / 'models/lr_24h_prc_v2_metadata.json') as f:
        prc_24h_features = json.load(f)['features']

    df = pd.read_parquet(PROJECT_ROOT / 'data/processed/model_ready.parquet')

    # Impute ORDC NaNs for LR models
    df_lr = df.copy()
    for col in ['RTORPA_log1p', 'RTOFFPA_log1p', 'RTORDPA_log1p']:
        if col in df_lr.columns:
            df_lr[col] = df_lr[col].fillna(0)

    # ── 1h Layer (decision-producing) ────────────────────────────
    results = predict_prc(df_lr.copy(), lr_1h, lgbm_1h, feature_cols=prc_1h_features)
    results = predict_price(results, price_1h, price_meta['features'],
                            price_meta['price_offset'])

    t_1h = thresholds['mining_1h']['tight_threshold']
    price_thresh = thresholds.get('price_1h', {}).get('mining_threshold')

    results['action'] = results.apply(
        lambda row: dispatch_action(
            row['predicted_prc'], row.get('predicted_price'),
            t_1h, price_thresh
        ), axis=1
    )

    print('--- 1h Layer (Decision) ---')
    print(results[['timestamp', 'predicted_prc', 'predicted_regime',
                   'predicted_price', 'action']].tail(24))

    # ── 4CP Layer (transmission charge avoidance) ────────────────
    results = predict_4cp(results, cp4, cp4_meta['features'])

    # Show 4CP alerts for summer afternoon hours
    summer_risk = results[(results['is_4cp_season'] == 1) & (results['p_4cp'] > 0.01)]
    if len(summer_risk) > 0:
        print(f'\n--- 4CP Risk ({len(summer_risk)} flagged hours) ---')
        print(summer_risk[['timestamp', 'p_4cp', 'load_total', 'temperature',
                           'action']].tail(20))
    else:
        print('\n--- 4CP: No high-risk hours in current data (outside summer or no risk) ---')

    # ── 24h Layer (advisory/planning) ────────────────────────────
    results_24h = predict_prc(df_lr.copy(), lr_24h, feature_cols=prc_24h_features)
    results_24h = predict_spread(results_24h, spread_24h, spread_meta['features'])

    t_24h = thresholds['mining_24h']['tight_threshold']
    results_24h['advisory'] = results_24h.apply(
        lambda row: advisory_24h(
            row['predicted_prc'], row.get('predicted_rt_24h'),
            row.get('DAM_price'), t_24h
        ), axis=1
    )

    print('\n--- 24h Layer (Advisory) ---')
    print(results_24h[['timestamp', 'predicted_prc', 'predicted_regime',
                       'DAM_price', 'predicted_spread', 'predicted_rt_24h',
                       'advisory']].tail(24))
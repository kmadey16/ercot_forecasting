# src/models/train.py
"""
Training script for PRC regression models (v2) and regime classifier (v1).

Regression (v2) — reverted to no sample weighting:
  Sample weights (50x/5x) degraded 1h MAE from 587→1703 MW by distorting LR's
  global fit toward rare scarcity conditions. Regression job: accurate PRC MW.
  Scarcity detection job: regime classifier (separate model).

Regime classifier (v1) — LGBMClassifier:
  Custom class weights (40/5/1/1), probability threshold tuning for Scarcity class.
  Trained on 2021-2023 only (all 169 scarcity events live here).
  Threshold chosen from out-of-fold CV probabilities to avoid optimism bias.
  2024 false positive audit: zero true scarcity in 2024, so any Scarcity
  prediction there is a false positive — measures generalization to modern grid.

PRC lags enabled (v2 classifier) — PRC_1h_lag, PRC_24h_lag, PRC_168h_lag added to feature set.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR    = PROJECT_ROOT / 'models'

BINS   = [0, 3000, 5000, 10000, float('inf')]
LABELS = ['Scarcity', 'Tight', 'Normal', 'Surplus']

# Dropped from all models: target, metadata, prices (backtesting only), raw adders
BASE_DROP = [
    'timestamp', 'PRC', 'regime',
    'RT_price', 'DAM_price', 'RT_DAM_spread',
    'RTORPA', 'RTOFFPA', 'RTORDPA',
]

# Additionally dropped from 1h model and classifier — PRC rolling stats add noise
# to models that already have PRC_1h_lag as a strong direct signal. They help the
# 24h model (trajectory info when current-hour actuals are unavailable).
DROP_1H = BASE_DROP + [
    'PRC_roll_mean_6h', 'PRC_roll_std_6h', 'PRC_roll_mean_24h',
]

# Additionally dropped from 24h model (not available 24h ahead)
DROP_24H = BASE_DROP + [
    'hub_load', 'load_total',
    'WGRPP_LZ_WEST', 'WGRPP_SYSTEM_WIDE', 'PVGRPP_SYSTEM_WIDE',
    'net_load_system', 'net_load_west',
    'renewable_pct_system', 'renewable_pct_west',
    'wind_forecast_error', 'solar_forecast_error',
    'RT_1h_lag', 'DAM_1h_lag',
    'RT_price_roll_mean_6h',  'RT_price_roll_std_6h',
    'DAM_price_roll_mean_6h', 'DAM_price_roll_std_6h',
    'hub_load_roll_mean_6h',  'hub_load_roll_std_6h',
    'load_total_roll_mean_6h','load_total_roll_std_6h',
    'WGRPP_LZ_WEST_roll_mean_6h', 'WGRPP_LZ_WEST_roll_std_6h',
    'load_total_ramp', 'hub_load_ramp',
    'wind_west_ramp', 'wind_system_ramp', 'RT_price_ramp',
    'RTORPA_log1p', 'RTOFFPA_log1p', 'RTORDPA_log1p',
]

# Classifier: custom weights — Scarcity 40x, Surplus 5x, rest baseline.
# Balanced would give Surplus 27x alongside Scarcity 38x, which wastes capacity
# on a class that's rare but highly learnable from features (high renewable + low load).
CLASSIFIER_WEIGHTS = {'Scarcity': 40, 'Tight': 1, 'Normal': 1, 'Surplus': 5}


def make_class_sample_weights(y_labels):
    """Convert class weight dict to per-sample weights.
    Passed to .fit(sample_weight=...) rather than the LGBMClassifier constructor
    because class_weight dict raises KeyError when a fold's training set doesn't
    contain all four classes (e.g. Surplus is absent from early-2021-only folds)."""
    return np.array([CLASSIFIER_WEIGHTS.get(str(label), 1.0) for label in y_labels])


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_data():
    return pd.read_parquet(PROCESSED_DIR / 'model_ready.parquet')


def impute_ordc_nans(df):
    """Fill 648 post-RTC+B NaN rows in ORDC log1p features with 0.
    Zero = no adder active, correct (no scarcity in that period).
    Always call on a copy — LGBM should see the raw NaNs."""
    for col in ['RTORPA_log1p', 'RTOFFPA_log1p', 'RTORDPA_log1p']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def build_X(df, drop_cols):
    return df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])


def save_with_metadata(model, pkl_path, metadata):
    joblib.dump(model, pkl_path)
    meta_path = pkl_path.parent / (pkl_path.stem + '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {pkl_path.name} + {meta_path.name}")


# ── Regression models (v2 — no sample weighting) ─────────────────────────────

def cv_scarcity_recall_regression(df, drop_cols, label):
    """Regression-based scarcity recall via TimeSeriesSplit within 2021-2023.
    No sample weighting — matches the reverted v2 regression approach."""
    cv_df = impute_ordc_nans(df[df['timestamp'] < '2024-01-01'].copy())
    X_cv  = build_X(cv_df, drop_cols).dropna()
    y_cv  = cv_df.loc[X_cv.index, 'PRC']

    tscv = TimeSeriesSplit(n_splits=4)
    fold_recalls = []

    print(f"\n--- Regression CV Scarcity Recall [{label}] ---")
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_cv)):
        X_tr, X_te = X_cv.iloc[tr_idx], X_cv.iloc[te_idx]
        y_tr, y_te = y_cv.iloc[tr_idx], y_cv.iloc[te_idx]

        lr_cv = LinearRegression()
        lr_cv.fit(X_tr, y_tr)
        residuals = y_tr - lr_cv.predict(X_tr)

        lgbm_cv = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm_cv.fit(X_tr, residuals)

        preds   = lr_cv.predict(X_te) + lgbm_cv.predict(X_te)
        actual  = pd.cut(y_te, bins=BINS, labels=LABELS)
        predicted = pd.cut(preds, bins=BINS, labels=LABELS)

        n_sc = (actual == 'Scarcity').sum()
        if n_sc > 0:
            report = classification_report(actual, predicted, output_dict=True, zero_division=0)
            recall = report.get('Scarcity', {}).get('recall', 0.0)
            fold_recalls.append(recall)
            print(f"  Fold {fold+1}: {n_sc} scarcity events | recall = {recall:.3f}")
        else:
            print(f"  Fold {fold+1}: 0 scarcity events — skipped")

    mean_recall = float(np.mean(fold_recalls)) if fold_recalls else 0.0
    print(f"  Mean CV recall: {mean_recall:.3f}")
    return mean_recall


def train_1h(df):
    print("\n" + "="*60)
    print("1h REGRESSION MODEL — v2 (no sample weighting)")
    print("="*60)

    df_lr = impute_ordc_nans(df.copy())
    train = df_lr[df_lr['timestamp'] < '2024-07-01']
    test  = df_lr[df_lr['timestamp'] >= '2025-01-01']

    valid   = build_X(train, DROP_1H).dropna().index
    X_train = build_X(train, DROP_1H).loc[valid]
    y_train = train.loc[valid, 'PRC']
    feature_cols = list(X_train.columns)

    X_test = build_X(test, DROP_1H)[feature_cols].dropna()
    y_test = test.loc[X_test.index, 'PRC']

    print(f"Features:   {len(feature_cols)}")
    print(f"Train:      {len(X_train):,} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"Test:       {len(X_test):,} rows   ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
    print(f"Train scarcity rows: {(y_train < 3000).sum()}")

    cv_recall = cv_scarcity_recall_regression(df.copy(), DROP_1H, '1h')

    print("\n--- Final model fit ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    residuals = y_train - lr.predict(X_train)
    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    lgbm.fit(X_train, residuals)

    preds = lr.predict(X_test) + lgbm.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.0f} MW")
    print(classification_report(pd.cut(y_test, bins=BINS, labels=LABELS),
                                pd.cut(preds,  bins=BINS, labels=LABELS), zero_division=0))

    metadata = {
        'model_version': 'v2',
        'training_date': str(date.today()),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'date_range_trained': {'start': str(train['timestamp'].min().date()),
                               'end':   str(train['timestamp'].max().date())},
        'cv_scarcity_recall': round(cv_recall, 4),
        'test_mae_mw': round(mae, 1),
        'sample_weights': 'none — reverted, see lgbm_regime_classifier_v1 for scarcity detection',
    }
    save_with_metadata(lr,   MODELS_DIR / 'lr_1h_prc_v2.pkl',        metadata)
    save_with_metadata(lgbm, MODELS_DIR / 'lgbm_1h_residual_v2.pkl', {**metadata, 'role': 'residual_corrector'})


def train_24h(df):
    print("\n" + "="*60)
    print("24h REGRESSION MODEL — v2 (no sample weighting)")
    print("="*60)

    df_lr = impute_ordc_nans(df.copy())
    train = df_lr[df_lr['timestamp'] < '2024-07-01']
    test  = df_lr[df_lr['timestamp'] >= '2025-01-01']

    valid   = build_X(train, DROP_24H).dropna().index
    X_train = build_X(train, DROP_24H).loc[valid]
    y_train = train.loc[valid, 'PRC']
    feature_cols = list(X_train.columns)

    X_test = build_X(test, DROP_24H)[feature_cols].dropna()
    y_test = test.loc[X_test.index, 'PRC']

    print(f"Features:   {len(feature_cols)}")
    print(f"Train:      {len(X_train):,} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"Test:       {len(X_test):,} rows   ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    cv_recall = cv_scarcity_recall_regression(df.copy(), DROP_24H, '24h')

    print("\n--- Final model fit ---")
    lr    = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.0f} MW")
    print(classification_report(pd.cut(y_test, bins=BINS, labels=LABELS),
                                pd.cut(preds,  bins=BINS, labels=LABELS), zero_division=0))

    metadata = {
        'model_version': 'v2',
        'training_date': str(date.today()),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'date_range_trained': {'start': str(train['timestamp'].min().date()),
                               'end':   str(train['timestamp'].max().date())},
        'cv_scarcity_recall': round(cv_recall, 4),
        'test_mae_mw': round(mae, 1),
        'sample_weights': 'none',
    }
    save_with_metadata(lr, MODELS_DIR / 'lr_24h_prc_v2.pkl', metadata)


# ── Regime classifier (v1) ────────────────────────────────────────────────────

def predict_with_threshold(probas, classes, threshold):
    """Override argmax for Scarcity: if P(Scarcity) >= threshold, predict Scarcity.
    Threshold < argmax default (0.25 for 4 classes) increases Scarcity recall
    at the cost of precision — threshold is the precision/recall dial."""
    scarcity_idx = list(classes).index('Scarcity')
    base_preds   = classes[np.argmax(probas, axis=1)]
    override     = probas[:, scarcity_idx] >= threshold
    return np.where(override, 'Scarcity', base_preds)


def cv_classifier_oof(df_window, feature_cols):
    """4-fold TimeSeriesSplit CV collecting out-of-fold probabilities.

    OOF approach: each row's probability comes from a model that never trained
    on that row. This gives an unbiased estimate for threshold selection —
    the same reason we use a holdout for hyperparameter tuning, not train error.

    Early folds (e.g. 2021-only training) may not see all 4 classes.
    We fix the global class order alphabetically and expand each fold's
    probability output to the full 4-column array, filling 0 for absent classes.
    """
    X = df_window[feature_cols].dropna()
    y = df_window.loc[X.index, 'regime'].astype(str)

    # Fixed global order — LGBM uses alphabetical by default
    all_classes = np.array(sorted(LABELS))
    tscv        = TimeSeriesSplit(n_splits=4)

    # Collect only test-fold rows — do NOT pre-allocate the full X array.
    # TimeSeriesSplit's first n rows are always in the training portion and
    # never appear in any test fold. Including them (with zero probabilities)
    # silently dilutes recall: Feb 2021 Uri events sit in that always-training
    # window and would be counted as missed predictions with P=0.
    oof_proba_list  = []
    oof_actual_list = []
    fold_meta       = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             random_state=42, verbose=-1)
        clf.fit(X_tr, y_tr, sample_weight=make_class_sample_weights(y_tr))

        # Expand to global 4-column layout (missing classes in fold → 0)
        fold_probas = np.zeros((len(te_idx), len(all_classes)))
        for i, cls in enumerate(clf.classes_):
            j = list(all_classes).index(cls)
            fold_probas[:, j] = clf.predict_proba(X_te)[:, i]

        oof_proba_list.append(fold_probas)
        oof_actual_list.extend(y_te.values)
        fold_meta.append({
            'fold':         fold + 1,
            'fold_probas':  fold_probas,
            'fold_actuals': y_te.values,
            'n_scarcity':   int((y_te == 'Scarcity').sum()),
            'n_rows':       len(te_idx),
        })

    oof_probas  = np.vstack(oof_proba_list)
    oof_actuals = np.array(oof_actual_list)
    return oof_probas, oof_actuals, all_classes, fold_meta


def find_scarcity_threshold(oof_probas, oof_actuals, classes, target_recall=0.65):
    """Sweep threshold 0.01→0.49, select highest threshold achieving target recall.
    Highest = best precision at the target recall level.
    If target is unreachable, report max achievable and return that threshold."""
    scarcity_idx = list(classes).index('Scarcity')

    sweep = []
    for t in np.arange(0.01, 0.50, 0.01):
        preds  = predict_with_threshold(oof_probas, classes, round(t, 2))
        report = classification_report(oof_actuals, preds, output_dict=True, zero_division=0)
        sc     = report.get('Scarcity', {})
        sweep.append({
            'threshold':    round(t, 2),
            'recall':       round(sc.get('recall',    0.0), 3),
            'precision':    round(sc.get('precision', 0.0), 3),
            'macro_recall': round(report.get('macro avg', {}).get('recall', 0.0), 3),
        })

    viable = [r for r in sweep if r['recall'] >= target_recall]
    chosen = max(viable, key=lambda r: r['threshold']) if viable else max(sweep, key=lambda r: r['recall'])

    print(f"\n--- Threshold sweep (target recall ≥ {target_recall}) ---")
    print(f"{'threshold':>10} {'sc_recall':>10} {'sc_prec':>9} {'macro_rec':>10}")
    for r in sweep:
        marker = " ◄" if r['threshold'] == chosen['threshold'] else ""
        print(f"  {r['threshold']:>8.2f}  {r['recall']:>9.3f}  {r['precision']:>8.3f}  {r['macro_recall']:>9.3f}{marker}")

    if not viable:
        print(f"\n  ⚠ Target recall {target_recall} not achievable. Max = {chosen['recall']:.3f} at threshold {chosen['threshold']}")
    else:
        print(f"\n  Chosen threshold: {chosen['threshold']} → recall={chosen['recall']}, precision={chosen['precision']}, macro_recall={chosen['macro_recall']}")

    return chosen, sweep


def per_fold_report(oof_probas, oof_actuals, classes, fold_meta, threshold):
    """Apply chosen threshold to each fold separately and report scarcity recall
    alongside macro recall — shows whether we're trading macro perf for scarcity."""
    print(f"\n--- Per-fold report at threshold={threshold} ---")
    print(f"{'Fold':>5} {'sc_events':>10} {'sc_recall':>10} {'macro_recall':>13}")

    for fm in fold_meta:
        fold_probas  = fm['fold_probas']
        fold_actuals = fm['fold_actuals']
        preds    = predict_with_threshold(fold_probas, classes, threshold)
        report   = classification_report(fold_actuals, preds, output_dict=True, zero_division=0)
        sc_rec   = report.get('Scarcity', {}).get('recall', 0.0)
        macro_rec = report.get('macro avg', {}).get('recall', 0.0)
        note     = " (no scarcity — skipped for recall)" if fm['n_scarcity'] == 0 else ""
        print(f"  {fm['fold']:>3}   {fm['n_scarcity']:>9}   {sc_rec:>9.3f}   {macro_rec:>12.3f}{note}")


def audit_false_positives_2024(model, df, feature_cols, threshold, classes):
    """2024 has zero true scarcity events — every Scarcity prediction is a false positive.
    Target: <2% of 2024 hours flagged as Scarcity."""
    df_2024 = impute_ordc_nans(
        df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] < '2025-01-01')].copy()
    )
    X_2024 = df_2024[feature_cols].dropna()

    probas = model.predict_proba(X_2024)
    preds  = predict_with_threshold(probas, classes, threshold)

    n_fp  = (preds == 'Scarcity').sum()
    pct   = n_fp / len(X_2024) * 100
    status = "✓ PASS" if pct < 2.0 else "✗ FAIL — too many false positives"

    print(f"\n--- 2024 False Positive Audit ---")
    print(f"  2024 rows evaluated: {len(X_2024):,}")
    print(f"  Predicted Scarcity:  {n_fp} ({pct:.2f}%)")
    print(f"  Target: <2.0%  →  {status}")

    # Show which hours were flagged (useful to check if they cluster oddly)
    if n_fp > 0 and n_fp <= 50:
        fp_ts = df_2024.loc[X_2024.index[preds == 'Scarcity'], 'timestamp']
        print(f"  Flagged hours: {sorted(fp_ts.dt.strftime('%Y-%m-%d %H:%M').tolist())}")

    return n_fp, round(pct, 2)


def train_regime_classifier(df):
    print("\n" + "="*60)
    print("REGIME CLASSIFIER — v2 (LGBMClassifier, PRC lags enabled)")
    print("="*60)
    print(f"Class weights: {CLASSIFIER_WEIGHTS}")

    # Classifier uses same 1h feature set + PRC_1h_lag, PRC_24h_lag, PRC_168h_lag.
    # Training window: 2021-2023 only — all 169 scarcity events live here.
    # Including 2024-2025 would add 17k rows with zero scarcity, diluting signal.
    df_clf = impute_ordc_nans(df[df['timestamp'] < '2024-01-01'].copy())
    feature_cols = list(build_X(df_clf, DROP_1H).dropna().columns)

    print(f"Features:         {len(feature_cols)}")
    print(f"Training window:  {df_clf['timestamp'].min().date()} → {df_clf['timestamp'].max().date()}")
    print(f"Total rows:       {len(df_clf):,}")
    print(f"Scarcity events:  {(df_clf['regime'] == 'Scarcity').sum()}")

    # Step 1: OOF CV to collect unbiased probabilities for threshold selection
    print("\n--- Running 4-fold OOF CV ---")
    oof_probas, oof_actuals, classes, fold_meta = cv_classifier_oof(df_clf, feature_cols)
    print(f"Classes: {list(classes)}")

    # Step 2: Sweep threshold on combined OOF predictions
    chosen, sweep = find_scarcity_threshold(oof_probas, oof_actuals, classes, target_recall=0.65)
    threshold = chosen['threshold']

    # Step 3: Per-fold breakdown at chosen threshold
    per_fold_report(oof_probas, oof_actuals, classes, fold_meta, threshold)

    # Step 4: Train final model on full 2021-2023 window
    print("\n--- Final model fit (full 2021-2023) ---")
    X_final = df_clf[feature_cols].dropna()
    y_final = df_clf.loc[X_final.index, 'regime'].astype(str)

    final_clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               random_state=42, verbose=-1)
    final_clf.fit(X_final, y_final, sample_weight=make_class_sample_weights(y_final))

    # Step 5: 2024 false positive audit (uses full df — outside training window)
    n_fp, fp_pct = audit_false_positives_2024(final_clf, df, feature_cols, threshold, final_clf.classes_)

    metadata = {
        'model_version': 'v2',
        'model_type': 'LGBMClassifier — 4-class regime, PRC lags enabled',
        'training_date': str(date.today()),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'date_range_trained': {
            'start': str(df_clf['timestamp'].min().date()),
            'end':   str(df_clf['timestamp'].max().date()),
        },
        'class_weights': CLASSIFIER_WEIGHTS,
        'scarcity_threshold': threshold,
        'cv_oof_scarcity_recall': chosen['recall'],
        'cv_oof_scarcity_precision': chosen['precision'],
        'cv_oof_macro_recall': chosen['macro_recall'],
        'false_positive_rate_2024_pct': fp_pct,
        'threshold_sweep': sweep,
    }
    save_with_metadata(final_clf, MODELS_DIR / 'lgbm_regime_classifier_v2.pkl', metadata)


# ── HB_WEST Price Model (Layer 2) ────────────────────────────────────────────

# Price target transform: log1p(RT_price + offset).
# Offset is derived from training data at runtime: abs(min RT price) + 1.0 buffer,
# so the most negative price maps to ~1.0 (safely above zero for log1p).
# log1p compresses the Uri tail ($5k→8.5) so LGBM doesn't overfit to extreme outliers
# and instead learns the $40-200 range where curtailment decisions actually happen.

# Dropped from price model: the target itself, closely related prices that would leak
# current-hour info, and raw ORDC adders (log1p versions kept).
# PRC and regime kept — they're valid input signals (Layer 1 output feeds Layer 2 context).
DROP_PRICE = [
    'timestamp', 'regime',
    'RT_price', 'DAM_price', 'RT_DAM_spread',
    'RTORPA', 'RTOFFPA', 'RTORDPA',
]


def train_price_model(df):
    """Train HB_WEST RT price model (LGBM regressor on log1p-transformed target).

    Layer 2 of the curtailment system: catches local congestion-driven price spikes
    that the PRC model (Layer 1) is blind to. 822 hours in 2021-2025 had Normal/Surplus
    PRC but >$100/MWh RT price — $61M exposure for a 200 MW miner.

    Split differs from PRC models: train Jan 2021–Dec 2023, val Jan–Jun 2024 (early
    stopping), test Jul 2024–Dec 2025. Price spikes are distributed across all years
    (no Uri-concentration problem), but severity is declining — test on recent conditions.
    """
    print("\n" + "="*60)
    print("HB_WEST PRICE MODEL — v1 (LGBM on log1p target)")
    print("="*60)

    # -- Split --
    train = df[df['timestamp'] < '2024-01-01'].copy()
    val   = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] < '2024-07-01')].copy()
    test  = df[df['timestamp'] >= '2024-07-01'].copy()

    # -- Build features (no dropna — LGBM handles NaN natively) --
    # Dropping NaN rows would lose ~3,500 rows from early 2021 where
    # TotalResourceMWZoneWest wasn't published yet.
    X_train = build_X(train, DROP_PRICE)
    X_val   = build_X(val,   DROP_PRICE)

    feature_cols = list(X_train.columns)
    X_val = X_val[feature_cols]
    X_test = build_X(test, DROP_PRICE)[feature_cols]

    # Only drop rows where the TARGET is NaN (can't train on missing target)
    train_mask = train.loc[X_train.index, 'RT_price'].notna()
    val_mask = val.loc[X_val.index, 'RT_price'].notna()
    test_mask = test.loc[X_test.index, 'RT_price'].notna()
    X_train, X_val, X_test = X_train[train_mask], X_val[val_mask], X_test[test_mask]

    # -- Build target (log1p-transformed) --
    y_train_raw = train.loc[X_train.index, 'RT_price']
    y_val_raw   = val.loc[X_val.index, 'RT_price']
    y_test_raw  = test.loc[X_test.index, 'RT_price']

    # Derive offset from training data so it adapts if future data has more negative prices.
    # +1.0 buffer so the minimum maps to ~1.0, not 0 (log1p(0)=0 would lose information).
    price_offset = float(np.abs(y_train_raw.min()) + 1.0)
    print(f"Price offset (derived from train min ${y_train_raw.min():.2f}): {price_offset:.2f}")

    y_train = np.log1p(y_train_raw + price_offset)
    y_val   = np.log1p(y_val_raw   + price_offset)
    y_test  = np.log1p(y_test_raw  + price_offset)

    print(f"Features:   {len(feature_cols)}")
    print(f"Train:      {len(X_train):,} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"Val:        {len(X_val):,} rows   ({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")
    print(f"Test:       {len(X_test):,} rows   ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
    print(f"Spikes >$100 — train: {(y_train_raw > 100).sum()}, val: {(y_val_raw > 100).sum()}, test: {(y_test_raw > 100).sum()}")

    # -- Train with early stopping on validation set --
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(50, verbose=True),
            __import__('lightgbm').log_evaluation(100),
        ],
    )
    print(f"Best iteration: {lgbm.best_iteration_}")

    # -- Predict and reverse transform --
    pred_log_test = lgbm.predict(X_test)
    pred_raw_test = np.expm1(pred_log_test) - price_offset

    # -- Metric 1: MAE on log1p scale --
    mae_log = mean_absolute_error(y_test, pred_log_test)
    print(f"\nTest MAE (log1p scale): {mae_log:.4f}")

    # -- Metric 2: MAE on raw $ by tier --
    print("\n--- MAE by price tier (raw $) ---")
    tiers = [
        ('Negative (<$0)',     y_test_raw < 0),
        ('Low ($0–40)',        (y_test_raw >= 0) & (y_test_raw <= 40)),
        ('Medium ($40–100)',   (y_test_raw > 40) & (y_test_raw <= 100)),
        ('High (>$100)',       y_test_raw > 100),
    ]
    for name, mask in tiers:
        if mask.sum() > 0:
            tier_mae = mean_absolute_error(y_test_raw[mask], pred_raw_test[mask.values])
            print(f"  {name:20s}  n={mask.sum():>5,}  MAE=${tier_mae:>8.2f}")

    # -- Metrics 3 & 4: Spike recall and precision at thresholds --
    print("\n--- Spike recall & precision ---")
    print(f"{'Threshold':>10} {'Actual':>7} {'Predicted':>10} {'Recall':>7} {'Precision':>10}")
    for thresh in [40, 100, 200]:
        actual_spike = (y_test_raw > thresh)
        pred_spike   = (pred_raw_test > thresh)
        n_actual     = actual_spike.sum()
        n_pred       = pred_spike.sum()
        tp           = (actual_spike.values & pred_spike).sum()
        recall       = tp / n_actual if n_actual > 0 else 0.0
        precision    = tp / n_pred if n_pred > 0 else 0.0
        print(f"  ${thresh:>7}  {n_actual:>6}  {n_pred:>9}  {recall:>6.3f}  {precision:>9.3f}")

    # -- Metric 5: Feature importance (top 15) --
    importance = pd.Series(lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Top 15 features (split importance) ---")
    for feat, imp in importance.head(15).items():
        print(f"  {feat:40s}  {imp:>5}")

    # -- Save --
    metadata = {
        'model_version': 'v1',
        'model_type': 'LGBMRegressor — HB_WEST RT price (log1p target)',
        'training_date': str(date.today()),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'price_offset': price_offset,
        'target_transform': f'log1p(RT_price + {price_offset})',
        'reverse_transform': f'expm1(pred) - {price_offset}',
        'date_range_train': {'start': str(train['timestamp'].min().date()),
                             'end':   str(train['timestamp'].max().date())},
        'date_range_val':   {'start': str(val['timestamp'].min().date()),
                             'end':   str(val['timestamp'].max().date())},
        'date_range_test':  {'start': str(test['timestamp'].min().date()),
                             'end':   str(test['timestamp'].max().date())},
        'best_iteration': lgbm.best_iteration_,
        'test_mae_log1p': round(mae_log, 4),
        'lgbm_params': {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
    }
    save_with_metadata(lgbm, MODELS_DIR / 'lgbm_price_1h_v1.pkl', metadata)

    return lgbm, feature_cols, importance


# ── 24h Price Model ──────────────────────────────────────────────────────────

# 24h spread model: predicts RT-DAM spread (how much RT will deviate from DAM).
# DAM price is KEPT as a feature (operator knows it 24h ahead — it already cleared).
# RT_price and RT_DAM_spread are dropped (target / derived from target).
# Everything else follows the same 24h-availability rules as before.
DROP_SPREAD_24H = [
    # Target components + metadata
    'timestamp', 'regime',
    'RT_price', 'RT_DAM_spread',
    'RTORPA', 'RTOFFPA', 'RTORDPA',
    # Not available 24h ahead: current-hour actuals
    'hub_load', 'load_total',
    'WGRPP_LZ_WEST', 'WGRPP_SYSTEM_WIDE', 'PVGRPP_SYSTEM_WIDE',
    'net_load_system', 'net_load_west',
    'renewable_pct_system', 'renewable_pct_west',
    'wind_forecast_error', 'solar_forecast_error',
    'wind_load_ratio_west',
    # Not available 24h ahead: 1h lags and short-horizon rolling stats
    'RT_1h_lag', 'DAM_1h_lag', 'PRC_1h_lag',
    'RT_DAM_spread_1h_lag', 'price_spike_lag',
    'RT_price_roll_mean_6h', 'RT_price_roll_std_6h',
    'DAM_price_roll_mean_6h', 'DAM_price_roll_std_6h',
    'hub_load_roll_mean_6h', 'hub_load_roll_std_6h',
    'load_total_roll_mean_6h', 'load_total_roll_std_6h',
    'WGRPP_LZ_WEST_roll_mean_6h', 'WGRPP_LZ_WEST_roll_std_6h',
    'PRC_roll_mean_6h', 'PRC_roll_std_6h',
    # Not available 24h ahead: ramp rates
    'load_total_ramp', 'hub_load_ramp',
    'wind_west_ramp', 'wind_system_ramp', 'RT_price_ramp',
    # ORDC adders: not available 24h ahead
    'RTORPA_log1p', 'RTOFFPA_log1p', 'RTORDPA_log1p',
]


def train_spread_model_24h(df):
    """Train 24h-ahead RT-DAM spread model.

    Reframed from absolute RT price prediction to spread prediction.
    Target: RT_price - DAM_price (raw dollars, no transform needed).

    Why spread instead of absolute price:
    - DAM price is already known 24h ahead (market cleared). No need to predict it.
    - The operator's question is "will RT exceed what DAM told me?" not "what will RT be?"
    - The spread is narrower and more symmetric than raw RT, easier to learn.
    - Matches how operators actually work: they have DAM positions and need to know
      if those positions are at risk.

    At inference: predicted_RT = DAM_price + predicted_spread.
    Curtailment rule: curtail if DAM_price + predicted_spread > threshold.

    DAM_price is a feature (not dropped) because the spread depends on price level:
    high-DAM hours during scarcity tend to have larger positive spreads.
    """
    print("\n" + "="*60)
    print("HB_WEST 24h SPREAD MODEL — v1 (RT-DAM spread, raw $)")
    print("="*60)

    # -- Split --
    train = df[df['timestamp'] < '2024-01-01'].copy()
    val   = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] < '2024-07-01')].copy()
    test  = df[df['timestamp'] >= '2024-07-01'].copy()

    # -- Build features (no dropna — LGBM handles NaN natively) --
    X_train = build_X(train, DROP_SPREAD_24H)
    X_val   = build_X(val,   DROP_SPREAD_24H)

    feature_cols = list(X_train.columns)
    X_val = X_val[feature_cols]
    X_test = build_X(test, DROP_SPREAD_24H)[feature_cols]

    # -- Build target: raw spread (RT - DAM) --
    # No transform — spread is roughly symmetric (median -$2.59), LGBM handles it well.
    # Extreme tails (Uri: +$5,891, -$8,478) exist but LGBM is robust to outliers.
    spread_train = train.loc[X_train.index, 'RT_price'] - train.loc[X_train.index, 'DAM_price']
    spread_val   = val.loc[X_val.index, 'RT_price'] - val.loc[X_val.index, 'DAM_price']
    spread_test  = test.loc[X_test.index, 'RT_price'] - test.loc[X_test.index, 'DAM_price']

    # Drop rows where spread can't be computed
    train_mask = spread_train.notna()
    val_mask = spread_val.notna()
    test_mask = spread_test.notna()
    X_train, X_val, X_test = X_train[train_mask], X_val[val_mask], X_test[test_mask]
    spread_train = spread_train[train_mask]
    spread_val = spread_val[val_mask]
    spread_test = spread_test[test_mask]

    # Keep raw prices for evaluation
    dam_test = test.loc[X_test.index, 'DAM_price']
    rt_test  = test.loc[X_test.index, 'RT_price']

    print(f"Features:   {len(feature_cols)}")
    print(f"Train:      {len(X_train):,} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"Val:        {len(X_val):,} rows   ({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")
    print(f"Test:       {len(X_test):,} rows   ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
    print(f"\nSpread stats (test):")
    print(f"  Median: ${spread_test.median():.2f}, Mean: ${spread_test.mean():.2f}")
    print(f"  Spread > $50: {(spread_test > 50).sum()},  > $100: {(spread_test > 100).sum()}")

    # -- Train with early stopping --
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train, spread_train,
        eval_set=[(X_val, spread_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(50, verbose=True),
            __import__('lightgbm').log_evaluation(100),
        ],
    )
    print(f"Best iteration: {lgbm.best_iteration_}")

    # -- Predict --
    pred_spread = lgbm.predict(X_test)
    pred_rt = dam_test.values + pred_spread  # reconstruct predicted RT

    # -- Metric 1: Spread MAE --
    spread_mae = mean_absolute_error(spread_test, pred_spread)
    print(f"\nSpread MAE: ${spread_mae:.2f}")

    # -- Metric 2: Reconstructed RT MAE by tier --
    rt_mae = mean_absolute_error(rt_test, pred_rt)
    print(f"Reconstructed RT MAE: ${rt_mae:.2f}")

    print("\n--- Reconstructed RT MAE by price tier ---")
    tiers = [
        ('Negative (<$0)',     rt_test < 0),
        ('Low ($0–40)',        (rt_test >= 0) & (rt_test <= 40)),
        ('Medium ($40–100)',   (rt_test > 40) & (rt_test <= 100)),
        ('High (>$100)',       rt_test > 100),
    ]
    for name, mask in tiers:
        if mask.sum() > 0:
            tier_mae = mean_absolute_error(rt_test[mask], pred_rt[mask.values])
            print(f"  {name:20s}  n={mask.sum():>5,}  MAE=${tier_mae:>8.2f}")

    # -- Metric 3: "Will RT exceed threshold?" recall/precision --
    # This is the actionable question: DAM says $X, will RT blow past $threshold?
    print("\n--- 'Will RT exceed threshold?' (using DAM + predicted spread) ---")
    print(f"{'Threshold':>10} {'Actual':>7} {'Predicted':>10} {'Recall':>7} {'Precision':>10}")
    for thresh in [40, 100, 200]:
        actual_spike = (rt_test > thresh).values
        pred_spike   = (pred_rt > thresh)
        n_actual     = actual_spike.sum()
        n_pred       = pred_spike.sum()
        tp           = (actual_spike & pred_spike).sum()
        recall       = tp / n_actual if n_actual > 0 else 0.0
        precision    = tp / n_pred if n_pred > 0 else 0.0
        print(f"  ${thresh:>7}  {n_actual:>6}  {n_pred:>9}  {recall:>6.3f}  {precision:>9.3f}")

    # -- Metric 4: Spread deviation buckets --
    # "How often does the model correctly flag large positive spreads?"
    print("\n--- Spread deviation prediction ---")
    print(f"{'Spread >':>10} {'Actual':>7} {'Predicted':>10} {'Recall':>7} {'Precision':>10}")
    for s_thresh in [20, 50, 100]:
        actual_dev = (spread_test > s_thresh).values
        pred_dev   = (pred_spread > s_thresh)
        n_actual   = actual_dev.sum()
        n_pred     = pred_dev.sum()
        tp         = (actual_dev & pred_dev).sum()
        recall     = tp / n_actual if n_actual > 0 else 0.0
        precision  = tp / n_pred if n_pred > 0 else 0.0
        print(f"  ${s_thresh:>7}  {n_actual:>6}  {n_pred:>9}  {recall:>6.3f}  {precision:>9.3f}")

    # -- Metric 5: Feature importance (top 15) --
    importance = pd.Series(lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Top 15 features (split importance) ---")
    for feat, imp in importance.head(15).items():
        print(f"  {feat:40s}  {imp:>5}")

    # -- Save --
    metadata = {
        'model_version': 'v1',
        'model_type': 'LGBMRegressor — HB_WEST 24h RT-DAM spread',
        'training_date': str(date.today()),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'target': 'RT_price - DAM_price (raw spread, no transform)',
        'inference': 'predicted_RT = DAM_price + predicted_spread',
        'date_range_train': {'start': str(train['timestamp'].min().date()),
                             'end':   str(train['timestamp'].max().date())},
        'date_range_val':   {'start': str(val['timestamp'].min().date()),
                             'end':   str(val['timestamp'].max().date())},
        'date_range_test':  {'start': str(test['timestamp'].min().date()),
                             'end':   str(test['timestamp'].max().date())},
        'best_iteration': lgbm.best_iteration_,
        'test_spread_mae': round(spread_mae, 2),
        'test_rt_mae_reconstructed': round(rt_mae, 2),
        'lgbm_params': {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
    }
    save_with_metadata(lgbm, MODELS_DIR / 'lgbm_spread_24h_v1.pkl', metadata)

    return lgbm, feature_cols, importance


# ── 4CP Peak Prediction Model ────────────────────────────────────────────────

# Features for 4CP model: everything available at prediction time.
# Uses load forecasts (known ahead), temperature, time features, and the
# running monthly max (observable up to current hour).
# Drops: RT_price and derivatives (4CP is about load, not price),
# plus all the standard target/metadata columns.
DROP_4CP = [
    'timestamp', 'PRC', 'regime',
    'RT_price', 'DAM_price', 'RT_DAM_spread',
    'RTORPA', 'RTOFFPA', 'RTORDPA',
    'RTORPA_log1p', 'RTOFFPA_log1p', 'RTORDPA_log1p',
    'RT_1h_lag', 'RT_24h_lag', 'RT_168h_lag',
    'RT_DAM_spread_1h_lag', 'price_spike_lag',
    'RT_price_roll_mean_6h', 'RT_price_roll_std_6h',
    'RT_price_roll_mean_24h', 'RT_price_roll_std_24h',
    'RT_price_ramp',
    'DAM_price_roll_mean_6h', 'DAM_price_roll_std_6h',
    'DAM_price_roll_mean_24h', 'DAM_price_roll_std_24h',
    # Must drop: target label and temp columns added during training
    'is_4cp_peak', 'year',
]


def train_4cp_model(df):
    """Train 4CP peak probability model.

    Predicts: P(this hour is the monthly coincident peak) for June-September.
    Target: binary (1 = this hour contained the monthly 4CP interval, 0 = it didn't).

    The model outputs a probability. Operators set a curtailment threshold based on
    their economics: how many afternoons of lost production are worth avoiding to
    dodge the ~$42K/MW/year transmission charge.

    Training data: summer hours only (June-Sept), 2021-2025.
    20 positive examples (4 months × 5 years) vs ~12,000 negative — extreme imbalance.
    Uses class weighting and probability calibration to handle this.

    The key feature is load_vs_monthly_max (current load / running monthly max).
    When this approaches or exceeds 1.0, the current hour is a strong 4CP candidate.
    """
    print("\n" + "="*60)
    print("4CP PEAK PROBABILITY MODEL — v1")
    print("="*60)

    # Load 4CP interval labels
    intervals = pd.read_csv(MODELS_DIR.parent / 'data' / 'raw' / '4cp_intervals_2021_2025.csv')
    intervals['timestamp'] = pd.to_datetime(intervals['timestamp'])
    print(f"4CP intervals loaded: {len(intervals)} peaks")

    # Filter to summer months only
    summer = df[df['month'].isin([6, 7, 8, 9])].copy()
    summer['year'] = summer['timestamp'].dt.year

    # Label: 1 if this hour matches a 4CP interval, 0 otherwise
    summer['is_4cp_peak'] = summer['timestamp'].isin(intervals['timestamp']).astype(int)
    print(f"Summer hours: {len(summer):,}")
    print(f"4CP peaks in data: {summer['is_4cp_peak'].sum()}")
    print(f"Class balance: {summer['is_4cp_peak'].mean():.5f} (1 in {int(1/summer['is_4cp_peak'].mean())})")

    # Split: train on 2021-2023 (12 peaks), test on 2024-2025 (8 peaks)
    train = summer[summer['year'] <= 2023].copy()
    test = summer[summer['year'] >= 2024].copy()

    X_train = build_X(train, DROP_4CP)
    X_test = build_X(test, DROP_4CP)
    feature_cols = list(X_train.columns)
    X_test = X_test[feature_cols]

    y_train = train['is_4cp_peak']
    y_test = test['is_4cp_peak']

    print(f"\nTrain: {len(X_train):,} rows ({train['year'].min()}-{train['year'].max()}), {y_train.sum()} peaks")
    print(f"Test:  {len(X_test):,} rows ({test['year'].min()}-{test['year'].max()}), {y_test.sum()} peaks")
    print(f"Features: {len(feature_cols)}")

    # Train with heavy class weighting — missing a peak is catastrophic ($2M+)
    # while false positives just cost a few hours of production
    scale_pos = int(len(y_train) / max(y_train.sum(), 1))
    print(f"Scale pos weight: {scale_pos}")

    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    # Predict probabilities on test set
    proba_test = lgbm.predict_proba(X_test)[:, 1]
    test_with_proba = test[['timestamp', 'year', 'month', 'load_total', 'temperature', 'hour']].copy()
    test_with_proba['p_4cp'] = proba_test
    test_with_proba['is_4cp_peak'] = y_test.values

    # -- Per-month analysis: did the model rank the actual peak highest? --
    print("\n--- Per-month peak detection (test set) ---")
    print(f"{'Year':>5} {'Month':>6} {'Peak hour':>20} {'P(4CP)':>8} {'Rank':>5} {'Top-1 correct':>14}")

    peaks_caught_top1 = 0
    peaks_caught_top3 = 0
    total_peaks = 0

    for (yr, mo), group in test_with_proba.groupby(['year', 'month']):
        actual_peak = group[group['is_4cp_peak'] == 1]
        if len(actual_peak) == 0:
            continue
        total_peaks += 1

        # Rank by probability
        ranked = group.sort_values('p_4cp', ascending=False).reset_index(drop=True)
        peak_rank = ranked[ranked['is_4cp_peak'] == 1].index[0] + 1
        peak_proba = actual_peak['p_4cp'].values[0]
        top1 = peak_rank == 1

        if top1:
            peaks_caught_top1 += 1
        if peak_rank <= 3:
            peaks_caught_top3 += 1

        peak_ts = actual_peak['timestamp'].values[0]
        print(f"  {yr:>4} {mo:>5}  {str(peak_ts)[:19]:>20} {peak_proba:>7.4f} {peak_rank:>5} {'✓' if top1 else '✗':>13}")

    print(f"\nPeaks caught (rank 1): {peaks_caught_top1}/{total_peaks}")
    print(f"Peaks caught (top 3):  {peaks_caught_top3}/{total_peaks}")

    # -- Threshold sweep: how many afternoons to curtail to catch all peaks? --
    print("\n--- Threshold sweep (test set) ---")
    print(f"{'Threshold':>10} {'Curtail hrs':>12} {'Peaks caught':>13} {'Missed':>7}")

    for thresh in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        curtail = (proba_test > thresh)
        caught = ((proba_test > thresh) & (y_test.values == 1)).sum()
        missed = y_test.sum() - caught
        print(f"  {thresh:>8.2f}  {curtail.sum():>11,}  {caught:>12}/{int(y_test.sum())}  {missed:>6}")

    # -- Economic backtest --
    print("\n--- Economic backtest (200 MW miner, $45K/MW/yr transmission) ---")
    CAPACITY = 200
    MINING_REV = 40
    TRANSMISSION_RATE = 45000  # $/MW/year

    for thresh in [0.05, 0.10, 0.20]:
        curtail_mask = proba_test > thresh
        curtail_hours = curtail_mask.sum()
        lost_production = curtail_hours * MINING_REV * CAPACITY  # revenue lost from curtailing

        # For each test year, check how many of the 4 peaks were dodged
        for yr in sorted(test['year'].unique()):
            yr_mask = test['year'].values == yr
            yr_proba = proba_test[yr_mask]
            yr_peaks = y_test.values[yr_mask]
            yr_curtail = yr_proba > thresh

            peaks_in_year = yr_peaks.sum()
            peaks_dodged = (yr_curtail & (yr_peaks == 1)).sum()
            peaks_missed = peaks_in_year - peaks_dodged

            # 4CP demand if missed peaks: (peaks_missed / 4) * capacity
            avg_4cp_demand = (peaks_missed / 4) * CAPACITY
            transmission_bill = avg_4cp_demand * TRANSMISSION_RATE
            curtail_hrs_yr = yr_curtail.sum()
            lost_prod_yr = curtail_hrs_yr * MINING_REV * CAPACITY

            print(f"  {yr} @{thresh:.2f}: curtail {curtail_hrs_yr:>4} hrs, "
                  f"dodge {peaks_dodged}/{peaks_in_year} peaks, "
                  f"transmission=${transmission_bill:>10,.0f}, "
                  f"lost production=${lost_prod_yr:>8,.0f}, "
                  f"net={'SAVE' if transmission_bill > lost_prod_yr else 'LOSE'} "
                  f"${abs(transmission_bill - lost_prod_yr):>10,.0f}")

    # -- Feature importance --
    importance = pd.Series(lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Top 15 features ---")
    for feat, imp in importance.head(15).items():
        print(f"  {feat:40s}  {imp:>5}")

    # -- Save --
    metadata = {
        'model_version': 'v1',
        'model_type': 'LGBMClassifier — 4CP peak probability',
        'training_date': str(date.today()),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'target': 'is_4cp_peak (binary: 1 = hour contains monthly coincident peak)',
        'training_window': 'June-Sept 2021-2023 (12 peaks)',
        'test_window': 'June-Sept 2024-2025 (8 peaks)',
        'scale_pos_weight': scale_pos,
        'peaks_caught_top1': peaks_caught_top1,
        'peaks_caught_top3': peaks_caught_top3,
        'total_test_peaks': total_peaks,
        'transmission_rate_assumption': TRANSMISSION_RATE,
        'lgbm_params': {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 5,
            'scale_pos_weight': scale_pos,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
    }
    save_with_metadata(lgbm, MODELS_DIR / 'lgbm_4cp_v1.pkl', metadata)

    return lgbm, feature_cols, importance


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_data()
    print(f"Loaded model_ready.parquet: {df.shape[0]:,} rows, {df.shape[1]} columns")
    train_1h(df)
    train_24h(df)
    train_regime_classifier(df)
    print("\nDone.")

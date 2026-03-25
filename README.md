# ERCOT Grid Reserve Forecasting for Flexible Load Dispatch

Predicts ERCOT Physical Responsive Capability (PRC) to guide curtailment decisions for Bitcoin mining and AI datacenter operations in West Texas.

## What This System Does

Forecasts grid reserve levels (PRC in MW) at 1-hour and 24-hour horizons, then uses economically optimized thresholds to recommend when flexible load operators should curtail. Evaluated against real-time prices at HB_WEST to quantify dollar savings.

**Three-layer architecture:**
- **Model Layer** Predicts PRC value. Stable across time periods and market redesigns.
- **Regime Labels** Translates PRC into human-readable grid state (Scarcity/Tight/Normal/Surplus). Fixed, physics-based boundaries for operator communication.
- **Decision Layer** Finds the economically optimal curtailment threshold. Adapts to changing grid conditions without retraining the model.

## What It Doesn't Do

This is a grid stress timing model, not a price forecaster. PRC predicts system-wide reserve levels, not local price. Congestion-driven price spikes at HB_WEST can occur during comfortable reserve levels. A price model conditioned on PRC would capture this — that's a planned enhancement.

## Results

### Model Performance

| Horizon | Model | Holdout MAE | CV Scarcity Recall | Use Case |
|---------|-------|-------------|-------------------|----------|
| 1h | LR + LGBM Ensemble | 587 MW | 41% | Real-time curtailment |
| 24h | Linear Regression | 1,359 MW | 8% | Next-day scheduling |

### Backtest — Economic Value (200 MW operation)

| Period | Use Case | Horizon | Savings | Hours Curtailed | Tight Threshold |
|--------|----------|---------|---------|-----------------|-----------------|
| 2025 | Mining | 1h | $1.0M | 166 / 8,401 | 6,750 MW |
| 2025 | Mining | 24h | $282K | 49 / 8,401 | 6,750 MW |
| 2025 | Datacenter | 1h | $35K | 40 / 8,401 | 6,250 MW |
| 2025 | Datacenter | 24h | $15K | 25 / 8,401 | 6,500 MW |
| 2022 | Mining | 1h | $25.3M | 1,764 / 2,594 | 4,750 MW |
| 2022 | Mining | 24h | $24.4M | 1,566 / 2,594 | 4,750 MW |
| 2022 | Datacenter | 1h | $7.4M | 1,572 / 2,594 | 4,500 MW |
| 2022 | Datacenter | 24h | $7.2M | 1,566 / 2,594 | 4,750 MW |

**Key insight:** Decision thresholds must be calibrated to current grid conditions. The model predicts PRC, and the decision layer adapts thresholds to the operating environment. 2022 tight threshold: 4,750 MW. 2025 tight threshold: 6,750 MW.

## Project Structure

```
ercot_forecasting/
├── data/
│   ├── raw/                          # Raw data files (not tracked)
│   ├── interim/                      # Cleaned individual datasets
│   └── processed/                    # model_ready.parquet
├── models/
│   ├── lr_1h_prc.pkl                 # 1h Linear Regression
│   ├── lgbm_1h_residual.pkl          # 1h LightGBM residual correction
│   ├── lr_24h_prc.pkl                # 24h Linear Regression
│   └── thresholds.json               # Optimized decision thresholds
├── notebooks/
│   ├── 01_eda_price_based.ipynb      # Original RT price exploration
│   ├── 01_eda_prc.ipynb              # PRC exploration and regime analysis
│   ├── 02_modeling_prc.ipynb         # PRC regression modeling + backtest
│   └── 03_modeling_price_based.ipynb # Original price-based classification
├── src/
│   ├── data/
│   │   ├── gridstatus_ingest.py      # Data ingestion (backfill + going-forward)
│   │   └── preprocess.py             # Cleaning and merging
│   ├── features/
│   │   └── feature_engineering.py    # Feature building pipeline
│   └── models/
│       ├── decision_layer.py         # Backtest and threshold optimization
│       └── predict.py                # Production prediction script
├── README.md
└── .gitignore
```

## How to Run

### 1. Data Pipeline

```bash
# Backfill historical prices (gridstatus open source)
python src/data/gridstatus_ingest.py

# Preprocess raw data → interim
python src/data/preprocess.py

# Build features → processed/model_ready.parquet
python src/features/feature_engineering.py
```

Historical Selenium-scraped data (2021-2025) is required in `data/raw/` for preprocessing. Post-RTC+B PRC data requires a GridStatus.io API key in `.env`.

### 2. Modeling

Run `notebooks/02_modeling_prc.ipynb` end-to-end. Saves trained models to `models/` and optimized thresholds to `models/thresholds.json`.

### 3. Prediction

```bash
python src/models/predict.py
```

Loads saved models and thresholds, predicts PRC for each hour, outputs dispatch recommendations.

## Modeling Approach

### Model Evolution

The project began with RT price forecasting at HB_WEST. Price proved too volatile — driven by bidding behavior, congestion, and market structure 
rather than predictable physical fundamentals. The pivot to PRC was motivated by two insights:

### Why PRC, Not Price
1. Price is a *result* of grid conditions, not a direct indicator of stress
2. ERCOT's December 2025 RTC+B market redesign changed the entire pricing 
   mechanism — price-based labels broke at the boundary, PRC did not.

The full progression is documented across the notebooks:

1. **RT Price Regression** (`01_eda_price_based.ipynb`, `02_modeling_price_based.ipynb`) — too volatile
2. **Price-Based Classification** (`02_modeling_price_based.ipynb`) — labels broke at RTC+B
3. **PRC Classification** — concept drift from grid capacity growth
4. **PRC Regression + Ensemble** (`02_modeling_prc.ipynb`) — final architecture

### Scarcity Detection

41% scarcity recall (1h model). Missed scarcity events are classified as Tight, not Normal — the model fails gracefully. Operators curtail during both Scarcity and Tight conditions, so the dispatch decision is often correct even when the regime label is wrong.

Scarcity detection evaluated via cross-validation on 2021-2023 data where events exist. The 2025 holdout has zero scarcity events due to grid capacity growth.

## Data Sources

- **gridstatus** (open source): load, wind, solar, outages, load forecasts, RT/DAM prices
- **GridStatus.io** (API): post-RTC+B PRC data (Dec 5, 2025+)
- **ERCOT archive xlsx**: pre-RTC+B PRC (2021 - Dec 4, 2025)
- **Open-Meteo API**: weather data for Midland, TX (West Texas hub)
- **Settlement point**: HB_WEST

## Production Roadmap

- **Threshold recalibration**: rerun `optimize_thresholds()` monthly/quarterly on recent data. Grid conditions evolve — what counts as economically worth curtailing shifts with capacity growth.
- **Model retraining**: retrain when MAE degrades significantly. PRC physics are stable but the distribution shifts as the grid grows.
- **Going-forward data**: `pull_new_data()` in `gridstatus_ingest.py` collects all datasets. Run weekly.
- **ORDC price adders**: post-RTC+B adder data (RTRDPA) collected but not yet integrated as features. Potential improvement for scarcity detection.

## Future Enhancements

- **Probability calibration**: isotonic regression on ensemble predictions for calibrated confidence intervals
- **Congestion features**: WESTEX binding constraints from `get_active_constraints()` to capture local price spikes independent of system-wide reserves
- **Battery state-of-charge tracking**: extend decision layer with charge/discharge cycles and efficiency losses
- **Structural capacity features**: rolling max wind/solar capacity as proxies for grid buildout trend
- **Price model conditioned on PRC**: separate model predicting RT price given PRC level, hour, and load conditions

## Tech Stack

Python, scikit-learn (Linear Regression), LightGBM, pandas, gridstatus, GridStatus.io API, Open-Meteo API, joblib
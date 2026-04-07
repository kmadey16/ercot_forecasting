# ERCOT Grid Forecasting for Flexible Load Dispatch

Predicts ERCOT grid reserves (PRC) and HB_WEST real-time electricity prices to guide curtailment decisions for Bitcoin mining, AI datacenter, and battery storage operations in West Texas.

## What This System Does

Two-layer forecasting system at 1-hour and 24-hour horizons:

- **Layer 1 (PRC):** Predicts system-wide Physical Responsive Capability (MW). Catches grid-wide emergencies — scarcity events, extreme weather, generation shortfalls.
- **Layer 2 (Price):** Predicts HB_WEST real-time electricity price ($/MWh). Catches local congestion-driven price spikes that occur independently of system-wide reserves.

Layers run in parallel. Curtailment triggers when **either** signal fires — covering both system-wide and local risk.

## Why Two Layers

822 hours in 2021-2025 had comfortable system-wide reserves (PRC Normal/Surplus) but HB_WEST prices above $100/MWh. Cost to a 200 MW miner: $61M. PRC-to-price correlation is only -0.116. West Texas is congestion-prone — local prices spike from transmission bottlenecks even when the grid overall is fine. A single PRC model is structurally blind to this.

## Results

### Model Performance

| Model | File | Key Metric |
|-------|------|------------|
| 1h PRC (LR + LGBM ensemble) | `lr_1h_prc_v2.pkl` + `lgbm_1h_residual_v2.pkl` | 538 MW MAE, 100% Uri scarcity recall |
| 1h Price (LGBM) | `lgbm_price_1h_v1.pkl` | 88.2% recall, 88.7% precision @ $100 |
| 24h PRC (LR) | `lr_24h_prc_v2.pkl` | 663 MW MAE |
| 24h Spread (LGBM) | `lgbm_spread_24h_v1.pkl` | RT-DAM spread, advisory only |

### Economic Backtest (Jul 2024 – Dec 2025)

| Use Case | Strategy | Savings (18 months) | % of Oracle |
|----------|----------|---------------------|-------------|
| Mining 200 MW | Combined @ $40 | $17.2M | 99.7% |
| Mining 200 MW | PRC-only | $22K | 0.1% |
| Datacenter 200 MW | Combined @ $60 | $2.2M | 103.7% |
| Oracle (perfect foresight) | — | $17.2M | 100% |

The price model provides nearly all economic value during normal operations. PRC is the safety net for rare catastrophic events (Winter Storm Uri-type).

### Uri Holdout Experiment

Models trained **without** Feb 2021, tested **on** Winter Storm Uri (most extreme event in ERCOT history):
- PRC regression: **100% scarcity recall** (72/72 hours caught)
- Price model: **99.6% recall** at $40 (missed 1 of 224 expensive hours)
- Combined system: **~$200M saved** for a 200 MW miner over 16 days
- Key insight: regression + thresholds is robust to unseen extremes; classification is fragile

### Post-RTC+B Validation (Jan–Apr 2026)

| Model | Training Period | Post-RTC+B | Status |
|-------|----------------|------------|--------|
| 1h PRC MAE | 538 MW | 877 MW | Degraded — retrain needed |
| 1h Price $100 recall | 88.2% | 92.7% | Holding strong |
| 1h Price $100 precision | 88.7% | 92.7% | Holding strong |

Price model works on new market structure. PRC model needs retraining with 2026 data — planned after agent system is built.

## Architecture

```
DATA SOURCES
  ├─ ERCOT MIS (CSV archives): load, wind, solar, outages, RT/DAM prices
  ├─ GridStatus.io API: post-RTC+B PRC + ORDC adders
  ├─ Open-Meteo: weather (Midland, TX — archive + forecast)
  ├─ EIA API: Waha Hub natural gas prices
  └─ gridstatus open-source: DAM system price (HB_HUBAVG)

PIPELINE
  ├─ src/data/gridstatus_ingest.py    → pull raw data
  ├─ src/data/preprocess.py           → clean/merge → interim/merged_all_data.parquet
  ├─ src/features/feature_engineering.py → 90 features → processed/model_ready.parquet
  └─ src/models/
      ├─ train.py                     → train all models
      ├─ predict.py                   → production scoring (1h decision + 24h advisory)
      └─ decision_layer.py            → backtest + threshold optimization

1h LAYER (DECISION — triggers curtailment)
  ├─ PRC regression: pred < 3,000 MW → CURTAIL | pred < 5,000 MW → REDUCE
  └─ Price regression: pred > $40 (mining) / $50 (datacenter) → REDUCE
      Either signal → take more aggressive action

24h LAYER (ADVISORY — informs scheduling)
  ├─ PRC regression: reserve risk assessment
  ├─ RT-DAM spread model: price deviation early warning
  └─ DAM price (free signal, no model needed)
```

## Use Cases

**Mining** (200 MW, binary on/off): 1h price model is the primary economic driver. Miners switch fast — real-time reaction is the decision driver. $17.2M/18mo backtest savings.

**Datacenter** (200 MW, 65% critical / 35% flexible): 24h layer (DAM + PRC) for workload pre-migration. 1h layer for real-time surprise response. $2.2M/18mo backtest savings.

**Battery** (planned): Different problem — not curtailment but charge/discharge/SoC optimization. Existing models are correct inputs; decision layer needs a state-of-charge optimizer.

## Data Sources

| Source | Type | Coverage | Key Data |
|--------|------|----------|----------|
| ERCOT MIS CSVs | Manual download | 2021–2026 | Load, wind, solar, outages, RT/DAM prices |
| GridStatus.io API | Automated pull | Post-RTC+B (Dec 2025+) | PRC, ORDC price adders |
| gridstatus (open source) | Automated pull | 2021–2026 | DAM system price (HB_HUBAVG) |
| Open-Meteo | API (free) | 2021–present | Temperature, humidity, wind, precipitation |
| EIA API | API (free key) | 2021–present | Waha Hub natural gas daily spot price |

## How to Run

```bash
# 1. Pull new data (weekly)
python -c "from src.data.gridstatus_ingest import pull_new_data; pull_new_data('2026-04-01', '2026-04-07')"

# 2. Preprocess
python src/data/preprocess.py

# 3. Build features
python src/features/feature_engineering.py

# 4. Score (production)
python -m src.models.predict

# 5. Train (when retraining needed)
python src/models/train.py
```

API keys required in `.env`: `GRIDSTATUS_API_KEY`, `EIA_API_KEY`.

## Key Design Decisions

- **Regression > Classification** for rare event detection. Regression extrapolates from continuous features; classifiers fail on unseen event types (validated via Uri holdout).
- **24h model predicts RT-DAM spread**, not absolute RT price. Operators already know DAM (it cleared day-ahead). The question is "will RT exceed DAM?" — a narrower, more achievable prediction.
- **log1p transform** on price target compresses extreme tail ($5,000 Uri prices → 8.5) so the model learns the $40–200 curtailment range, not just outliers.
- **Price offset derived from data** (not hardcoded) — adapts on retrain if future data has more negative prices.
- **PRC rolling stats** help 24h model (trajectory info) but hurt 1h model (multicollinearity with PRC_1h_lag). Excluded per-model.

## Roadmap

- [x] Layer 1: PRC models (1h + 24h)
- [x] Layer 2: Price models (1h + 24h spread)
- [x] Economic backtest (5-strategy comparison)
- [x] Uri holdout validation
- [x] Post-RTC+B data pipeline (GridStatus.io + ERCOT MIS)
- [ ] Multi-agent system (OpenClaw) — data/feature/model/synthesis agents
- [ ] Deployment (always-on server + Telegram alerts)
- [ ] MLOps (automated retraining, drift detection)
- [ ] Battery use case (AS price model + SoC optimizer)

## Tech Stack

Python 3.12, scikit-learn, LightGBM, pandas, gridstatus, GridStatus.io API, Open-Meteo API, EIA API, joblib

# ERCOT Grid Stress & Price Spike Prediction System

**822 hours of $100+ electricity prices that the grid said were fine. This system catches them.**

ERCOT's Physical Responsive Capability (PRC) measures system-wide reserve levels — but West Texas (HB_WEST) prices spike from local congestion even when reserves are comfortable. A 200 MW Bitcoin miner paying $100+/MWh during those hours loses $61M over 5 years. This system predicts both grid-wide stress and local price spikes, triggering curtailment before the cost hits.

## The Edge

Most approaches predict either grid reserves or electricity price. Neither alone works:

- **PRC-only** catches system-wide emergencies (Uri, statewide heat) but misses congestion. During our 18-month test window, PRC-only saved **$22K**. The price model saved **$17.2M**. During this window there were zero system-wide scarcity events — congestion-driven spikes were the entire story. PRC is the safety net for Uri-type events; price is the daily P&L driver.
- **Price-only** catches congestion but can't see system-wide emergencies 24h ahead.

This system runs both in parallel. Either signal triggers curtailment. In backtest: **99.7% of perfect-foresight savings within this test window** for a 200 MW mining operation.

## How It Makes Money

### Economic Backtest (Jul 2024 – Apr 2026, 14,801 hours)

**Mining (200 MW, $40/MWh revenue baseline):**

| Strategy | What it is | Savings | % Oracle |
|----------|-----------|---------|----------|
| DAM-only @$40 | Read DAM prices, curtail when high | $20.0M | 83.8% |
| RT lag-react @$40 | Watch last hour's price, react | $21.3M | 89.0% |
| **Combined system @$40** | **Our 1h PRC + price models** | **$23.8M** | **99.7%** |
| Oracle | Perfect foresight | $23.9M | 100% |

**Datacenter (200 MW, 65% critical / 35% flexible, $50 penalty):**

| Strategy | Savings | % Oracle |
|----------|---------|----------|
| DAM-only @$60 | $2.5M | 67.3% |
| RT lag-react @$60 | $2.7M | 73.7% |
| **Combined system @$60** | **$3.2M** | **85.9%** |
| Oracle @>$143 | $3.7M | 100% |

**Incremental value vs what operators already do:**
- vs DAM-only: **+$3.8M** more over 18 months (19% improvement)
- vs reacting to last hour's RT price: **+$2.5M** more (12% improvement)
- Combined system has **169 false positive hours** ($53K lost) vs **1,008 false positives** for DAM-only ($1.7M lost) — 6x fewer bad calls

Assumes signal received at hour start, curtailment executable within 15 minutes, no ramp constraints modeled.

### Winter Storm Uri — Unseen Event Detection

Models trained **without** Feb 2021 (removed from training data), tested **on** Uri:

- PRC model: **100% scarcity recall** — caught all 72 hours of grid emergency
- Price model: **99.6% recall at $40** — missed 1 of 224 expensive hours
- Combined: **~$200M saved** for a 200 MW miner in 16 days (400:1 savings-to-cost ratio)

The models detected an event type they had never seen. Regression + thresholds extrapolates from continuous features — classification fails on novel extremes (the classifier scored 0/72).

## System Architecture

```
1h LAYER — DECISION (triggers curtailment)
  ├─ PRC model:   pred < 3,000 MW → CURTAIL  |  pred < 5,000 MW → REDUCE
  └─ Price model: pred > $40/MWh  → REDUCE (mining)
                  pred > $50/MWh  → REDUCE (datacenter)
      ↳ Either signal fires → take the more aggressive action

4CP LAYER — TRANSMISSION (June-Sept, avoids annual charges)
  └─ 4CP model:   P(peak day) > threshold → CURTAIL afternoon window
      ↳ Reduces curtailment from ~50 afternoons to ~10 (work in progress)

24h LAYER — ADVISORY (informs scheduling, no auto-curtailment)
  ├─ PRC model:         next-day reserve risk
  ├─ RT-DAM spread:     price deviation early warning
  └─ DAM price (free):  strongest 24h input, no model needed
```

The 1h layer drives real-time curtailment. The 4CP layer drives summer transmission avoidance. The 24h layer informs planning.

## Model Performance

| Model | Type | Key Metric | Role |
|-------|------|------------|------|
| 1h PRC | LR + LGBM ensemble | 538 MW MAE, 100% Uri recall | Safety net for grid emergencies |
| 1h Price | LGBM regressor | 88% recall @ $100, 89% precision | Primary economic driver |
| 4CP Risk | LGBM classifier | Day-level peak risk (in progress) | Transmission charge avoidance |
| 24h PRC | Linear Regression | 663 MW MAE | Next-day reserve planning |
| 24h Spread | LGBM regressor | Predicts RT-DAM deviation | Early warning (advisory) |

**Post-RTC+B validation (Jan–Apr 2026):** Price model holds strong (92.7% recall at $100) on new market structure. PRC degraded (877 vs 538 MW MAE) — retrain scheduled, market redesign shifted reserve dynamics.

## Use Cases

**Bitcoin Mining** (200 MW, binary on/off, $40/MWh revenue): The 1h price model is the primary real-time decision (+$3.8M vs DAM-only baseline). 4CP avoidance (~$9M/year potential) is in development — 12 training examples makes reliable ML prediction challenging; conservative rule-based approach currently more cost-effective.

**AI Datacenter** (200 MW, 65% critical / 35% flexible, $50/MWh SLA penalty): Two-layer workflow — read DAM prices + 24h PRC to pre-migrate flexible workloads overnight, then let the 1h layer handle real-time surprises. **+$700K vs DAM-only** over 18 months.

**Battery Storage** (planned): Not curtailment — arbitrage. Buy low, sell high, bid ancillary services. Same model inputs, different decision layer (state-of-charge optimization).

## Key Design Decisions

**Regression > Classification for rare events.** Regression extrapolates from continuous features (PRC dropping → predict lower). Classification needs to have seen the pattern. Validated via Uri holdout: regression caught 72/72 scarcity hours; classifier caught 0/72.

**24h model predicts RT-DAM spread, not absolute price.** DAM clears day-ahead — operators already know it. The question is "will RT exceed DAM?" not "what will RT be?" Reframing tripled recall (34.9% vs 11.5% at $100).

**log1p target transform.** RT prices range -$39 to $5,000. Without compression, the model overfits to Uri-level outliers and underfits the $40–200 range where curtailment decisions happen.

**Thresholds optimized per use case.** Mining threshold = $40 (revenue breakeven). Datacenter threshold = $50 (SLA penalty breakeven). Different economics → different optimal points.

## Data Pipeline

Six data sources, automated ingestion, 97 engineered features:

```
ERCOT MIS (CSV)     → load, wind, solar, outages, RT/DAM prices (2021–2026)
GridStatus.io API   → PRC, ORDC price adders (post-RTC+B, Dec 2025+)
gridstatus (open)   → DAM system price for congestion spread
Open-Meteo API      → weather (temperature, humidity, wind, precipitation)
EIA API             → Waha Hub natural gas daily spot price
ERCOT xlsx archives → pre-RTC+B PRC + ORDC adders (2021–Nov 2025)
```

## Databricks Deployment

The full pipeline runs on Databricks with medallion architecture:

```
BRONZE (raw ingestion)
  └─ CSV/xlsx/parquet files → Delta tables via Unity Catalog

SILVER (cleaning + transformation)
  ├─ 01_bronze_to_silver_ingest    → raw files to Delta tables
  └─ 02_silver_cleaning_backfill   → timestamp alignment, HB_WEST filtering,
                                      hour-ending fixes, ORDC/PRC extraction,
                                      weather API pull, deduplication

GOLD (ML-ready)
  ├─ 03_gold_merge                 → join all silver tables on timestamp
  ├─ 04_gold_features              → 97 features (lags, rolling stats, spreads,
  │                                   regime labels, 4CP indicators)
  └─ ercot.gold.model_ready        → final Delta table for training/scoring

MODELS (MLflow Registry)
  ├─ 05_model_training             → train all models, log to MLflow
  │   ├─ ercot-prc-1h-lr           → 1h PRC Linear Regression
  │   ├─ ercot-prc-1h-lgbm        → 1h PRC LGBM residual
  │   ├─ ercot-prc-24h            → 24h PRC Linear Regression
  │   ├─ ercot-price-1h           → 1h Price LGBM
  │   ├─ ercot-spread-24h         → 24h RT-DAM Spread LGBM
  │   └─ ercot-4cp-peak           → 4CP peak classifier (in progress)
  ├─ 06_backtest                   → realistic baseline comparison
  └─ 07_scoring                    → production scoring pipeline
        └─ ercot.gold.predictions  → scored output Delta table
```

Models are versioned in MLflow with parameters, metrics, and artifacts tracked per run. Scoring loads models from the registry and writes predictions back to Delta.

## How to Run

**On Databricks:** Run notebooks 01→07 in sequence, or schedule as a Workflow.

**Locally:**
```bash
python -c "from src.data.gridstatus_ingest import pull_new_data; pull_new_data('2026-04-01', '2026-04-07')"
python src/data/preprocess.py
python src/features/feature_engineering.py
python -m src.models.predict
```

Requires `.env` with `GRIDSTATUS_API_KEY` and `EIA_API_KEY`.

## Roadmap

- [x] Layer 1: PRC models (1h + 24h) — grid-wide reserve forecasting
- [x] Layer 2: Price models (1h absolute + 24h spread) — local congestion detection
- [ ] 4CP peak prediction — $9M/year potential, limited by 12 training examples and settlement-adjusted load gap
- [x] Economic backtest: realistic baselines (DAM-only, RT lag-react) with dollar-denominated savings
- [x] Extreme event validation: Uri holdout proving robustness to unseen events
- [x] Post-RTC+B pipeline: handles ERCOT market redesign (Dec 2025) seamlessly
- [x] **Databricks migration** — Delta Lake medallion architecture, MLflow model registry, 7 notebooks end-to-end
- [ ] **Databricks Workflows** — scheduled orchestration of the pipeline
- [ ] **Lakehouse Monitoring** — automated drift detection on predictions table
- [ ] **Multi-agent system** — autonomous data/feature/model/synthesis agents
- [ ] **Battery arbitrage** — AS price model + state-of-charge optimizer

## Tech Stack

Python 3.12 | Databricks (Delta Lake, MLflow, Unity Catalog) | scikit-learn | LightGBM | PySpark | pandas | gridstatus | GridStatus.io API | Open-Meteo API | EIA API

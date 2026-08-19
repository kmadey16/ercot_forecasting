# ERCOT Grid Stress & Price Spike Prediction System

**822 hours of $100+ electricity prices that the grid said were fine. This system catches them.**

ERCOT's Physical Responsive Capability (PRC) measures system-wide reserve levels — but West Texas (HB_WEST) prices spike from local congestion even when reserves are comfortable. A 200 MW Bitcoin miner paying $100+/MWh during those hours loses $61M over 5 years. This system predicts both grid-wide stress and local price spikes, triggering curtailment before the cost hits.

## The Edge

Most approaches predict either grid reserves or electricity price. Neither alone works:

- **PRC-only** catches system-wide emergencies (Uri, statewide heat) but misses congestion. During our 18-month test window there were **zero system-wide scarcity events** — congestion-driven spikes were the entire story, which is why the *price* model (not the reserve model) drives the economics. PRC remains the safety net for Uri-type tail events.
- **Price-only** catches congestion but can't see system-wide emergencies 24h ahead.

This system runs both in parallel. Either signal triggers curtailment. In backtest, the combined system captures **99.7% of perfect-foresight savings** within the test window for a 200 MW mining operation.

## How It Makes Money

### Economic Backtest (Jul 2024 – Apr 2026, 14,801 hours)

Baselines reflect what operators actually do — not an always-on strawman.

**Mining (200 MW, $40/MWh revenue baseline):**

| Strategy                 | What it is                         | Savings    | % Oracle  |
| ------------------------ | ---------------------------------- | ---------- | --------- |
| DAM-only @$40            | Read DAM prices, curtail when high | $20.0M     | 83.8%     |
| RT lag-react @$40        | Watch last hour's price, react     | $21.3M     | 89.0%     |
| **Combined system @$40** | **Our 1h PRC + price models**      | **$23.8M** | **99.7%** |
| Oracle                   | Perfect foresight                  | $23.9M     | 100%      |

**Datacenter (200 MW, 65% critical / 35% flexible, $50 penalty):**

| Strategy                 | Savings   | % Oracle  |
| ------------------------ | --------- | --------- |
| DAM-only @$60            | $2.5M     | 67.3%     |
| RT lag-react @$60        | $2.7M     | 73.7%     |
| **Combined system @$60** | **$3.2M** | **85.9%** |
| Oracle @>$143            | $3.7M     | 100%      |

**Incremental value vs. what operators already do (mining):**

- vs DAM-only: **+$3.8M** more over 18 months (19% improvement)
- vs reacting to last hour's RT price: **+$2.5M** more (12% improvement)
- Combined system has **169 false-positive hours** ($53K lost) vs **1,008 false positives** for DAM-only ($1.7M lost) — **6x fewer bad calls**, 839 fewer hours of unnecessary curtailment

*Assumes signal received at hour start, curtailment executable within 15 minutes, no ramp constraints modeled.*

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

4CP LAYER — TRANSMISSION (investigational, June–Sept)
  └─ Daily peak model: flags likely monthly-peak days to narrow curtailment
      ↳ Not production-ready — see 4CP note below

24h LAYER — ADVISORY (informs scheduling, no auto-curtailment)
  ├─ PRC model:         next-day reserve risk
  ├─ RT-DAM spread:     price deviation early warning
  └─ DAM price (free):  strongest 24h input, no model needed
```

The 1h layer drives real-time curtailment. The 24h layer informs planning. The 4CP layer is an active investigation, not a shipped model (details below).

## Model Performance

| Model      | Type               | Key Metric                          | Status                          |
| ---------- | ------------------ | ----------------------------------- | ------------------------------- |
| 1h PRC     | LR + LGBM ensemble | 538 MW MAE, 100% Uri recall         | Production — safety net         |
| 1h Price   | LGBM regressor     | 88% recall @ $100                   | Production — primary driver     |
| 24h PRC    | Linear Regression  | 663 MW MAE                          | Production (advisory)           |
| 24h Spread | LGBM regressor     | 34.9% recall @ $100 (RT-DAM spread) | Production (advisory)           |
| 4CP Daily  | LGBM classifier    | 2/8 peaks caught                    | **Not production-ready**        |

**Post-RTC+B validation (Jan–Apr 2026):** Price model holds strong (92.7% recall at $100) on the new market structure. PRC degraded (877 vs 538 MW MAE) — retrain scheduled, since the market redesign shifted reserve dynamics.

## 4CP (Four Coincident Peak) — Honest Status

4CP transmission charges are real and large (~$9M/year for a 200 MW load, set by just four 15-minute intervals per summer), so predicting the peak intervals is economically attractive. It is also genuinely hard, and **this remains an open investigation, not a shipped capability:**

- An initial hourly model showed **7/7 peaks caught — which turned out to be target leakage** (`is_4cp_peak` accidentally left in the features). After removing it: **1/7**.
- A reframed day-level model reached **2/8 peaks caught** at usable thresholds — only ~12 positive training examples exist across the training years, too few for reliable ML.
- ERCOT's official 4CP uses settlement-adjusted load only finalized after month-end, so the exact target isn't reproducible in real time.

**Current recommendation:** a hybrid approach — the model narrows ~50 candidate afternoons to ~15, and the operator curtails the flagged afternoons — which captures most of the savings without over-trusting an unreliable classifier. Lesson banked: always verify suspiciously perfect results for leakage before believing them.

## Key Design Decisions

**Regression > Classification for rare events.** Regression extrapolates from continuous features (PRC dropping → predict lower); classification needs to have seen the pattern. Validated via Uri holdout: regression caught 72/72 scarcity hours; the classifier caught 0/72.

**24h model predicts RT-DAM spread, not absolute price.** DAM clears day-ahead — operators already know it. The question is "will RT exceed DAM?" not "what will RT be?" Reframing tripled recall (34.9% vs 11.5% at $100).

**log1p target transform.** RT prices range -$39 to $5,000. Without compression, the model overfits Uri-level outliers and underfits the $40–200 range where curtailment decisions actually happen.

**Thresholds optimized per use case.** Mining threshold = $40 (revenue breakeven); datacenter threshold = $50 (SLA penalty breakeven). Different economics → different optimal points.

**Realistic baselines, not strawmen.** The original backtest compared against always-on (no curtailment) — no real operator runs that way. Rebuilt against DAM-only and RT-lag-react baselines so the reported value reflects what the system adds over standard operator practice.

## Data Pipeline — Databricks Lakehouse

A medallion architecture on Databricks: **Bronze and Silver run as a Delta Live Tables (Lakeflow Declarative) pipeline; the Gold layer is built in dbt.** 14 source feeds (across ~6 providers) flow through 33 pipeline tables into one model-ready table.

- **Bronze** — Auto Loader (`cloudFiles`) incrementally ingests parquet from a Unity Catalog volume, with schema evolution (`addNewColumns`) and ingestion metadata (`_ingested_at`, source path) stamped on every row.
- **Silver** — typed, deduplicated, HB_WEST-filtered tables with per-source RTC+B cutover handling. Dual-format feeds (wind, solar, load forecast, DAM system lambda, weather) are normalized across ERCOT-CSV and open-source schemas, then written via `AUTO CDC` flows as **SCD Type 1** — keyed on timestamp, sequenced by publish vintage — so late-arriving revisions correctly supersede earlier values.
- **Gold (dbt)** — staging → intermediate (`int_merged_all`, `int_model_ready`) → marts, with tests, macros, and documentation; produces the **45K+ row hourly modeling table** (~90 production features) that feeds the models.

Feeds: RT/DAM prices, PRC, zonal load, ORDC price adders, ancillary-service prices, DAM system lambda, wind, solar, outages, load forecast, DAM hub average, Waha gas, and weather — spanning ERCOT MIS, GridStatus.io, gridstatus (open), Open-Meteo, and EIA.

## How to Run

```
# Pull new data (weekly)
python -c "from src.data.gridstatus_ingest import pull_new_data; pull_new_data('2026-04-01', '2026-04-07')"

# Preprocess → features → score
python src/data/preprocess.py
python src/features/feature_engineering.py
python -m src.models.predict

# Retrain (when drift detected)
python src/models/train.py
```

Requires `.env` with `GRIDSTATUS_API_KEY` and `EIA_API_KEY`.

## Roadmap

- [x] Layer 1: PRC models (1h + 24h) — grid-wide reserve forecasting
- [x] Layer 2: Price models (1h absolute + 24h spread) — local congestion detection
- [x] Economic backtest: realistic baselines (DAM-only, RT lag-react) with dollar-denominated savings
- [x] Extreme-event validation: Uri holdout proving robustness to unseen events
- [x] Post-RTC+B pipeline: handles ERCOT market redesign (Dec 2025)
- [~] 4CP peak prediction — investigated; hybrid narrowing approach viable, standalone model not production-ready
- [x] **Databricks Lakehouse** — DLT medallion pipeline (Auto Loader, schema evolution, SCD-1 CDC) + dbt gold layer, on Unity Catalog
- [ ] **Multi-agent system** — autonomous data/feature/model/synthesis agents
- [ ] **Deployment** — always-on server + real-time alerts
- [ ] **MLOps** — automated retraining, drift detection, model versioning
- [ ] **Battery arbitrage** — AS price model + state-of-charge optimizer

## Tech Stack

Databricks (Delta Live Tables / Lakeflow Declarative Pipelines, Unity Catalog, Auto Loader, Delta Lake) | dbt (staging / intermediate / marts, tests) | PySpark | Python 3.12 | scikit-learn | LightGBM | pandas | gridstatus | GridStatus.io / Open-Meteo / EIA APIs

## About

Predicting ERCOT grid stress conditions to optimize flexible load dispatch for datacenters, bitcoin miners, and battery storage.
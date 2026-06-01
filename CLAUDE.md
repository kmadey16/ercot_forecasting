# ERCOT Forecasting Project

Predicts ERCOT Physical Responsive Capability (PRC) to guide curtailment decisions for Bitcoin mining and AI datacenter operations in West Texas (HB_WEST hub).

## Architecture

Three-layer pipeline:

1. **Data ingestion** (`src/data/gridstatus_ingest.py`) — pulls from gridstatus (open), GridStatus.io API (PRC post-Dec 5 2025), and Open-Meteo (weather). Run `pull_new_data()` weekly.
2. **Preprocessing** (`src/data/preprocess.py`) — cleans/merges 10+ sources into `data/interim/merged_all_data.parquet`. Aggregates 15-min RT prices to hourly, handles HB_WEST filtering, unifies pre/post RTC+B PRC sources.
3. **Feature engineering** (`src/features/feature_engineering.py`) — builds 50+ features (lags, rolling stats, forecast errors, net load, cyclical time) → `data/processed/model_ready.parquet`.
4. **Models** (`src/models/`) — 1h ensemble (LR + LGBM residual) and 24h LR, trained on PRC target. Decision layer backtests thresholds, `predict.py` is the production scoring script.

## Primary Goal
Transform this project from a static ML model into a production-grade MLOps + multi-agent system:

1. **MLOps layer** — automated retraining pipeline, model versioning, drift detection, 
   performance monitoring, threshold recalibration
2. **Multi-agent system (OpenClaw)** — wrap existing models in an agent architecture:
   - Data agent: pulls, validates, and refreshes ERCOT/weather data
   - Feature agent: runs preprocessing and feature engineering pipeline
   - Model agent: scores current conditions, flags regime, confidence intervals
   - Synthesis agent: combines outputs into actionable curtailment recommendation
   - Orchestrator: coordinates agents, handles failures, routes to Telegram
3. **Deployment** — planned: always-on local server (Mac Mini, not yet purchased), 
   Telegram interface for real-time alerts (not yet set up). 
   Build the agent system first, deployment infrastructure comes later.

### Success Criteria
- Scarcity recall ≥ 0.65 (1h model)
- System runs autonomously with minimal manual intervention
- Curtailment recommendations delivered via Telegram before dispatch window
- Portfolio-ready for DRW AI Engineer application (fall 2026)

## Communication Style
- I am learning as I build this project — explain what you're doing and why as you go
- When writing new code, briefly explain the design decision before writing it
- When debugging, explain what the bug was and why the fix works
- Don't just fix things silently — teach me what's happening
- Flag anything that could break existing models or data pipelines before doing it
- Also note any design decisions i could make to improve this, or other data/strategies that could help improve it

## Key Data

- **Target**: PRC (MW) — ERCOT grid reserve proxy, stable across market redesigns
- **Horizon**: 1h (real-time dispatch) and 24h (next-day scheduling)
- **Hub**: HB_WEST (West Texas)
- **Timeframe**: Jan 2021 – present (hourly)
- **PRC regimes**: Scarcity (<3k MW), Tight (3–5k MW), Normal (5–10k MW), Surplus (>10k MW)
- **Critical break**: Dec 5, 2025 — ERCOT RTC+B market redesign. PRC source switches from ERCOT xlsx archives to GridStatus.io API.

## Models

### Layer 1: PRC Models (system-wide reserve forecasting)

| File | Horizon | Type | MAE | CV Scarcity Recall |
|------|---------|------|-----|-----|
| `models/lr_1h_prc_v2.pkl` | 1h | Linear Regression | 538 MW | 0.380 |
| `models/lgbm_1h_residual_v2.pkl` | 1h | LGBM (residual correction) | — | — |
| `models/lr_24h_prc_v2.pkl` | 24h | Linear Regression | 663 MW | 0.263 |
| `models/lgbm_regime_classifier_v2.pkl` | 1h | LGBM Classifier | — | 0.236 (OOF) |

v2 models retrained 2026-03-31 on corrected data (RT double-shift fix, DAM gap fill, weather dual-source).
Previous v1 models had inflated MAE due to 1-hour RT price misalignment.

### Layer 2: HB_WEST Price Model (local congestion-driven spike detection)

| File | Horizon | Type | Test MAE (log1p) | $100 Recall | $100 Precision |
|------|---------|------|-------------------|-------------|----------------|
| `models/lgbm_price_1h_v1.pkl` | 1h | LGBM Regressor (absolute RT) | 0.0193 | 0.882 | 0.887 |
| `models/lgbm_spread_24h_v1.pkl` | 24h | LGBM Regressor (RT-DAM spread) | $12.06 spread MAE | 0.349 | 0.465 |

**1h model:** Target `log1p(RT_price + offset)`, 82 features, 499 iterations.
Top features: RT_price_ramp, RT_1h_lag, RT_DAM_spread_1h_lag.

**24h model (reframed):** Target `RT_price - DAM_price` (raw spread). 46 features, 4 iterations.
Predicts "how much will RT deviate from DAM?" — DAM price is a known feature, not a prediction target.
At inference: `predicted_RT = DAM_price + predicted_spread`.
Recall tripled vs absolute price model (34.9% vs 11.5% at $100) because DAM does the heavy lifting —
when DAM is already high, even a small predicted spread flags risk correctly.
Top features: PRC, DAM_price, RT_price_roll_std_24h, DAM_system, days_since_start.

Data sources added: DAM congestion spread (HB_HUBAVG from gridstatus), Waha Hub gas prices (EIA API),
West zone outage capacity (reinstated from existing data).
Split: train Jan 2021–Dec 2023, val Jan–Jun 2024, test Jul 2024–Dec 2025.

**Why Layer 2 exists:** 822 hours (2021-2025) where PRC was Normal/Surplus but HB_WEST RT price >$100/MWh.
PRC-to-price correlation is only -0.116. West Texas prices spike from local congestion independently of
system-wide reserves. Layers run in parallel: curtail when EITHER signal triggers.

### Thresholds (`models/thresholds.json`)

**PRC thresholds:** Scarcity <2,000 MW, Tight <6,750 MW. Must be recalibrated as grid capacity grows.

**Price thresholds (optimized 2026-04-06 on Jan–Jun 2024 val set):**
- Mining: $40/MWh — any hour above mining revenue baseline is a losing hour
- Datacenter: $50/MWh — matches curtailment penalty breakeven

### Economic Backtest Results (Jul 2024 – Dec 2025, 12,144 hours)

**Mining (200 MW, binary curtailment, $40/MWh revenue baseline):**

| Strategy | Savings vs Always-on | % of Oracle | Hours Curtailed |
|----------|---------------------|-------------|-----------------|
| PRC-only | $22,313 | 0.1% | 2 |
| Combined @$40 | $17,161,585 | 99.7% | 2,772 |
| Combined @$60 | $14,731,805 | 85.6% | 1,117 |
| Combined @$100 | $9,231,309 | 53.6% | 331 |
| Oracle | $17,215,309 | 100% | 2,716 |

False positives at $40: 118 hours, $31,754 lost revenue (0.18% of savings).

**Datacenter (200 MW, 65% critical / 35% flexible, $50/MWh penalty):**

| Strategy | Savings vs Always-on | % of Oracle |
|----------|---------------------|-------------|
| PRC-only | $3,205 | 0.2% |
| Combined @$60 | $2,187,116 | 103.7% |
| Combined @$100 | $1,499,629 | 71.1% |
| Oracle @>$143 | $2,109,275 | 100% |

Datacenter @$60 beats oracle (103.7%) because partial load shedding at moderate spikes
is more efficient than oracle's all-or-nothing shed at $143.

**Key finding:** During this test period (no system-wide scarcity events), the price model
provides nearly all economic value. PRC-only saves ~$22K (mining) / ~$3K (datacenter).
Combined system saves $9–17M (mining) / $1.5–2.2M (datacenter). PRC remains essential as
a safety net for rare catastrophic events (Uri-type).

Backtest notebook: `notebooks/03_price_model_backtest.ipynb`

## Use Cases

- **Mining** (200 MW, binary on/off, $40/MWh revenue baseline):
  - Primary driver: 1h price model ($17M+ / 18mo). Miners can flip a switch in minutes, so real-time reaction is the decision driver.
  - 1h PRC: safety net for system-wide emergencies. 24h: read DAM for awareness, not a curtailment trigger.
  - Backtest savings: ~$17.2M/18mo (1h combined @$40), ~$10.8M (24h+1h combined @$100).

- **Datacenter** (200 MW, 65% critical / 35% flexible, $50/MWh curtailment penalty):
  - **24h layer (planning):** Read DAM prices + 24h PRC forecast → pre-migrate flexible workloads away from expensive hours.
  - **1h layer (reactive):** 1h price + 1h PRC → catch surprise congestion spikes, shed flexible load in real-time.
  - 24h handles predictable risk, 1h handles surprises. Together ~100% of oracle at $60 threshold.
  - Backtest savings: ~$2.3M/18mo (1h @$60), ~$2.0M (24h+1h combined @$80).

- **Battery** (not yet built):
  - Different problem: batteries WANT high prices (to discharge into). Value is buy-low-sell-high arbitrage.
  - Needs: SoC optimization, AS price model (ECRS/RRS), co-optimization across charge/discharge/hold/AS bid.
  - Existing models are correct inputs, but decision layer is an optimization problem, not threshold-based.

## Environment

- Python 3.12.3 (`.python-version`)
- Venv at `venv/` — activate with `source venv/bin/activate`
- API key in `.env`: `GRIDSTATUS_API_KEY`
- No `requirements.txt` — key packages: pandas, numpy, scikit-learn, lightgbm, joblib, gridstatus, gridstatusio, python-dotenv, requests, pytz

## Notebooks

- `notebooks/01_eda_prc.ipynb` — active EDA (PRC regimes, data quality)
- `notebooks/02_modeling_prc.ipynb` — active modeling (ensemble, backtest, threshold optimization)
- `notebooks/03_price_model_backtest.ipynb` — Layer 2 price model 5-strategy economic backtest (mining + datacenter)
- `*_price_based.ipynb` — legacy (abandoned after RTC+B broke price models)

## Data Files (not tracked in git)

- `data/raw/` — CSV/xlsx source files
- `data/interim/*.parquet` — cleaned per-source datasets
- `data/processed/model_ready.parquet` — final ML-ready dataset

## Limitations / Planned Work

- PRC predicts system-wide reserve level, not local price. Congestion-driven spikes at HB_WEST during comfortable reserves are not captured.
- Planned: price model conditioned on PRC to capture congestion effects.
- Retrain when holdout MAE degrades; recalibrate thresholds quarterly.
- Add in economic backtest for battery storage use case

## What NOT to do
- Do not modify raw data files in data/raw/
- Do not overwrite model .pkl files without explicit instruction
- Do not retrain models mid-session unless asked
- Do not touch legacy price-based notebooks

## Current Model Status (updated 2026-06-01)
- **Layer 1 (PRC):** v2 models retrained 2026-03-31 on corrected data
  - 1h regression: 538 MW MAE, 0.380 CV scarcity recall
  - 24h regression: 663 MW MAE, 0.263 CV scarcity recall
  - Classifier v2 (PRC lags): 0.236 OOF scarcity recall, 0.00% 2024 FP rate
- **Layer 2 (Price):** v1 built 2026-04-06
  - 1h LGBM: 0.0168 log1p MAE, 89.7% recall / 89.4% precision at $100 threshold
  - Backtest: Combined system saves $17.2M/18mo (mining @$40) / $2.2M/18mo (datacenter @$60)
  - PRC-only saves <$23K in same period — price model provides nearly all economic value during non-scarcity periods
- **Price thresholds optimized:** mining $40, datacenter $50 (saved to thresholds.json)
- **predict.py updated:** dispatch_action() now checks both PRC and price signals in parallel
- **Features added:** RT_DAM_spread_1h_lag, wind_load_ratio_west, price_spike_lag, DAM_congestion_spread, DAM_system, TotalResourceMWZoneWest (reinstated)
- model_ready.parquet: regenerated 2026-04-06, 89 columns ✓
- Pre/post RTC+B flag: added (is_pre_rtcb) ✓
- ORDC price adders: added as features (RTORPA_log1p, RTOFFPA_log1p, RTORDPA_log1p) ✓
- PRC lags: ENABLED (PRC_1h_lag, PRC_24h_lag, PRC_168h_lag) ✓
- **24h model reframed:** Spread model (RT-DAM) replaces absolute price model. Recall tripled (34.9% vs 11.5% at $100).
- **New data sources:** Waha Hub gas prices (EIA API), DAM system price (HB_HUBAVG), West zone outages (reinstated)
- model_ready.parquet: regenerated 2026-04-06, 90 columns ✓
- **predict.py updated:** 1h (decision) + 24h (advisory) + 4CP (transmission) layers wired end-to-end
- **4CP model (2026-06-01):** LGBM classifier predicting P(4CP peak day). Initial attempt had target leakage (fake 7/7 result). After fix: 2/8 peaks caught at usable thresholds. 12 training examples limits ML reliability. Day-level model with 15-min load data is best attempt so far. Conservative rule-based approach (curtail ~50 afternoons) currently more cost-effective.
- model_ready.parquet: regenerated 2026-06-01, 97 columns (7 new 4CP features) ✓
- **Uri holdout experiment (2026-04-06):** Models trained WITHOUT Feb 2021, tested ON Uri.
  PRC regression: 100% scarcity recall (72/72). Price model: 99.6% recall at $40, 79.1% at $1000.
  Combined system would have curtailed 271/361 hours, saving ~$200M on a 200 MW miner.
  Classifier failed completely (0/72 recall) — no scarcity examples in training.
  Key insight: regression + thresholds is robust to unseen events; classifiers are fragile.

## Classifier Goal (reframed 2026-03-30)
The 0.65 scarcity recall target was set before understanding the training data structure.

**The Uri problem:** 114 of 169 scarcity events are permanently in the always-training partition
(TimeSeriesSplit never puts them in a test fold). All 72 Uri hours are in that partition.
OOF recall is measured on 55 non-Uri events only — summer heat spikes and isolated cold snaps.
Uri recall cannot be measured via this CV setup. A separate holdout experiment is required.

**Reframed goals:**
1. Maximize recall on non-Uri summer heat events (the recurring operational risk — fold 2's
   44 events represent the pattern that actually repeats year-over-year)
2. Preserve Uri signal — model should still train on Uri, just not measured via OOF CV
3. Build a separate Uri holdout test: train on non-Uri data, test on Feb 2021, measure recall
   on the most extreme event type independently

**Next feature work:** Rolling PRC stats (PRC_roll_mean_6h, PRC_roll_std_6h, PRC_roll_mean_24h)
to capture build-up patterns — the rate of PRC decline may be more informative than point-in-time
lags for distinguishing genuine scarcity onset from brief dips.

## Domain Context
- ERCOT Physical Responsive Capability (PRC) is the system-wide operating reserve margin
- Scarcity conditions (<3k MW PRC) trigger ancillary service price spikes
- RTC+B (Dec 5 2025) redesigned real-time co-optimization — PRC data source changed, 
  pre/post period must be treated carefully in training data
- HB_WEST pricing is congestion-prone, can spike independently of system-wide PRC

## NOTE
- PRC lags are NOW ENABLED in feature_engineering.py (PRC_1h_lag, PRC_24h_lag, PRC_168h_lag).
  The previous "do not uncomment" note is superseded — lags were enabled 2026-03-30
  after v1 classifier baseline was established. model_ready.parquet has 80 columns.

import pandas as pd
import numpy as np


# ── Mining backtest ──────────────────────────────────────────────────────────

def mining_backtest(test_df, predicted_prc, tight_threshold, scarcity_threshold=3000, capacity_mw = 200, mining_revenue_per_mwh=40,verbose=True):
     #mining_rev_per_mwh: $/MWh mining income
     #capacity_mw: miner size in MW

     results = test_df[['timestamp', 'RT_price']].copy()
     results['predicted_prc'] = predicted_prc

     #Cost for always-on
     results['always_on_cost'] = (results['RT_price'] - mining_revenue_per_mwh) * capacity_mw

     #Model guided suggestions curtailing during tight/scarcity
     results['guided_cost'] = results['always_on_cost'].copy()


     results.loc[results['predicted_prc'] < scarcity_threshold, 'guided_cost'] = 0
     results.loc[(results['predicted_prc'] >= scarcity_threshold) & (results['predicted_prc'] < tight_threshold), 'guided_cost'] = 0

     #savings
     total_always_on = results['always_on_cost'].sum()
     total_guided = results['guided_cost'].sum()
     savings = total_always_on - total_guided

     hours_curtailed = (results['guided_cost'] == 0).sum()

     if verbose:
          print(f'Always-on cost: ${total_always_on:,.0f}')
          print(f'Model-guided cost: ${total_guided:,.0f}')
          print(f'Savings: ${savings:,.0f}')
          print(f'Hours curtailed: {hours_curtailed} / {len(results)}')

     return results


# ── Datacenter backtest ──────────────────────────────────────────────────────

def datacenter_backtest(test_df, predicted_prc, tight_threshold=5000,scarcity_threshold=3000,critical_pct=0.65, capacity_mw=200, curtailment_penalty=50, verbose=True):
     #Critical_pct: percentage of critical load
     #capacity_mw:  datacenter size in MW

     results = test_df[['timestamp', 'RT_price']].copy()
     results['predicted_prc'] = predicted_prc

     flexible_capacity_pct = 1 - critical_pct

     # Always full power
     results['always_on_cost'] = results['RT_price'] * capacity_mw

     # Model guided using prob
     results['guided_load'] = capacity_mw  # default full power
     results.loc[results['predicted_prc'] < scarcity_threshold,'guided_load'] = capacity_mw * critical_pct  # cut all flexible load (scarcity)
     results.loc[(results['predicted_prc'] >= scarcity_threshold) & (results['predicted_prc'] < tight_threshold),'guided_load'] = capacity_mw * (critical_pct + flexible_capacity_pct * 0.5)  # cut half flexible (tight)

     reduced_mw = capacity_mw - results['guided_load']

     results['guided_cost'] = (results['RT_price'] * results['guided_load']) + (curtailment_penalty * reduced_mw)

     total_always_on = results['always_on_cost'].sum()
     total_guided = results['guided_cost'].sum()

     savings = total_always_on - total_guided

     if verbose:
          print(f'Always-on cost: ${total_always_on:,.0f}')
          print(f'Model-guided cost: ${total_guided:,.0f}')
          print(f'Savings: ${savings:,.0f}')
          print(f'Hours at reduced load: {(results["guided_load"] < capacity_mw).sum()} / {len(results)}')

     return results


# ── PRC threshold optimizer ──────────────────────────────────────────────────

def optimize_thresholds(val_df, predicted_prc, backtest_fn, capacity_mw=200, verbose=False, **kwargs):
     best_savings = 0
     best_params = {}

     for s_threshold in range(2000, 4500, 250):
          for t_threshold in range(4000,9000,250):
               if t_threshold <= s_threshold:
                    continue
               results = backtest_fn(val_df, predicted_prc, capacity_mw=capacity_mw, scarcity_threshold=s_threshold, tight_threshold=t_threshold,verbose=verbose, **kwargs)
               savings = results['always_on_cost'].sum() - results['guided_cost'].sum()

               if savings > best_savings:
                    best_savings = savings
                    best_params = {'tight_threshold': t_threshold, 'scarcity_threshold': s_threshold}

     return best_params, best_savings


# ── Price threshold optimizer ────────────────────────────────────────────────
#
# Sweeps candidate price thresholds on a validation set to find the one that
# maximizes net savings. For each threshold, it computes:
#   - Avoided cost: money saved by curtailing hours where actual RT > threshold
#   - Lost revenue: money forfeited by curtailing hours where actual RT < revenue baseline
#   - Net = avoided - lost
#
# The threshold with the highest net savings wins. This is use-case-specific:
# mining (binary on/off) and datacenter (partial shed) have different economics.

def optimize_price_threshold_mining(val_df, predicted_price, capacity_mw=200,
                                    mining_revenue_per_mwh=40, verbose=True):
    """Find optimal price curtailment threshold for mining.

    Mining is binary: when predicted price exceeds threshold, shut off entirely.
    Net cost of staying on = (RT_price - mining_revenue) * capacity.
    Curtailing saves that cost when RT is high, but loses (mining_rev - RT) * capacity
    when RT was actually low (false positive).
    """
    best_thresh = None
    best_net = -np.inf
    sweep = []

    for thresh in np.arange(30, 125, 5):
        curtail = predicted_price > thresh
        actual_rt = val_df['RT_price'].values

        # Hours curtailed where RT > mining_rev: we correctly avoided a loss
        true_pos = curtail & (actual_rt > mining_revenue_per_mwh)
        avoided = ((actual_rt[true_pos] - mining_revenue_per_mwh) * capacity_mw).sum()

        # Hours curtailed where RT <= mining_rev: we lost mining revenue unnecessarily
        false_pos = curtail & (actual_rt <= mining_revenue_per_mwh)
        lost = ((mining_revenue_per_mwh - actual_rt[false_pos]) * capacity_mw).sum()

        net = avoided - lost
        sweep.append({
            'threshold': float(thresh),
            'hours_curtailed': int(curtail.sum()),
            'true_positives': int(true_pos.sum()),
            'false_positives': int(false_pos.sum()),
            'avoided_cost': float(avoided),
            'lost_revenue': float(lost),
            'net_savings': float(net),
        })

        if net > best_net:
            best_net = net
            best_thresh = float(thresh)

    if verbose:
        print("\n--- Price threshold sweep (mining) ---")
        print(f"{'thresh':>7} {'curtail':>8} {'tp':>5} {'fp':>5} {'avoided':>12} {'lost':>10} {'net':>12}")
        for r in sweep:
            marker = " ◄" if r['threshold'] == best_thresh else ""
            print(f"  ${r['threshold']:>5.0f} {r['hours_curtailed']:>7} {r['true_positives']:>5} "
                  f"{r['false_positives']:>5} ${r['avoided_cost']:>10,.0f} ${r['lost_revenue']:>8,.0f} "
                  f"${r['net_savings']:>10,.0f}{marker}")
        print(f"\n  Best: ${best_thresh:.0f} → net savings ${best_net:,.0f}")

    return best_thresh, best_net, sweep


# ── Multi-strategy comparison backtest ───────────────────────────────────────
#
# Compares our system against realistic operator baselines:
# - DAM-only: read day-ahead prices, curtail when high (free, no model)
# - RT lag-react: watch last hour's RT price, react (someone watching a screen)
# - Our system: ML model predicting ahead
# - Oracle: perfect foresight
#
# This replaces the old "vs always-on" comparison with what operators actually do.

def compare_strategies_mining(test_df, pred_price, pred_prc,
                              price_threshold=40, prc_tight=5000,
                              capacity_mw=200, mining_revenue_per_mwh=40,
                              verbose=True):
    """Compare curtailment strategies against realistic baselines for mining.

    Mining is binary on/off. Net cost per hour = (RT_price - mining_rev) * capacity.
    Curtailing sets cost to 0 (no power, no mining, no bill).
    """
    bt = test_df[['timestamp', 'RT_price', 'DAM_price']].copy()
    bt['net_cost'] = (bt['RT_price'] - mining_revenue_per_mwh) * capacity_mw
    bt['pred_price'] = pred_price
    bt['pred_prc'] = pred_prc

    total_always_on = bt['net_cost'].sum()
    oracle_cost = bt['net_cost'].clip(upper=0).sum()
    oracle_savings = total_always_on - oracle_cost

    def calc(curtail_mask, name):
        cost = bt['net_cost'].where(~curtail_mask, 0).sum()
        savings = total_always_on - cost
        hrs = curtail_mask.sum()
        fp = (curtail_mask & (bt['RT_price'] <= mining_revenue_per_mwh)).sum()
        fp_cost = ((mining_revenue_per_mwh - bt.loc[
            curtail_mask & (bt['RT_price'] <= mining_revenue_per_mwh), 'RT_price'
        ]) * capacity_mw).sum()
        pct = savings / oracle_savings * 100 if oracle_savings > 0 else 0
        return {'strategy': name, 'savings': savings, 'hours_curtailed': hrs,
                'false_positives': fp, 'fp_cost': fp_cost, 'pct_oracle': pct}

    strategies = []
    strategies.append({'strategy': 'Always-on', 'savings': 0,
                       'hours_curtailed': 0, 'false_positives': 0,
                       'fp_cost': 0, 'pct_oracle': 0})

    # Baselines: what operators actually do
    for t in [40, 60, 80]:
        strategies.append(calc(bt['DAM_price'] > t, f'DAM-only @${t}'))

    for t in [40, 60, 80]:
        lag_react = bt['RT_price'].shift(1).fillna(0) > t
        strategies.append(calc(lag_react, f'RT lag-react @${t}'))

    # Our models
    strategies.append(calc(bt['pred_prc'] < prc_tight, 'PRC model'))

    for t in [price_threshold]:
        strategies.append(calc(bt['pred_price'] > t, f'Price model @${t}'))
        combined = (bt['pred_prc'] < prc_tight) | (bt['pred_price'] > t)
        strategies.append(calc(combined, f'Combined system @${t}'))

    strategies.append({'strategy': 'Oracle', 'savings': oracle_savings,
                       'hours_curtailed': (bt['RT_price'] > mining_revenue_per_mwh).sum(),
                       'false_positives': 0, 'fp_cost': 0, 'pct_oracle': 100})

    results = pd.DataFrame(strategies)

    if verbose:
        print(f"\n{'Strategy':<25} {'Savings':>12} {'Hours':>7} {'FP hrs':>7} "
              f"{'FP cost':>10} {'% Oracle':>9}")
        print('-' * 75)
        for _, r in results.iterrows():
            print(f"{r['strategy']:<25} ${r['savings']:>10,.0f} {r['hours_curtailed']:>7,} "
                  f"{r['false_positives']:>7,} ${r['fp_cost']:>8,.0f} "
                  f"{r['pct_oracle']:>8.1f}%")

        # Incremental value vs baselines
        combined = results[results['strategy'].str.startswith('Combined')].iloc[0]
        print(f"\n--- Incremental value of our system ---")
        for baseline in ['DAM-only @$40', 'DAM-only @$60', 'RT lag-react @$40']:
            base_row = results[results['strategy'] == baseline]
            if not base_row.empty:
                inc = combined['savings'] - base_row.iloc[0]['savings']
                print(f"  vs {baseline:<22} +${inc:>10,.0f}")

    return results


def compare_strategies_datacenter(test_df, pred_price, pred_prc,
                                  price_threshold=60, prc_tight=5000,
                                  prc_scarcity=3000, capacity_mw=200,
                                  critical_pct=0.65, curtailment_penalty=50,
                                  verbose=True):
    """Compare curtailment strategies against realistic baselines for datacenter.

    Datacenter sheds half of flexible load when triggered. Cost = electricity for
    reduced load + penalty on shed capacity.
    """
    bt = test_df[['timestamp', 'RT_price', 'DAM_price']].copy()
    bt['pred_price'] = pred_price
    bt['pred_prc'] = pred_prc

    flex_mw = capacity_mw * (1 - critical_pct)
    load_half_shed = capacity_mw - (flex_mw * 0.5)
    cost_always_on = (bt['RT_price'] * capacity_mw).sum()

    # Oracle: shed all flexible when RT > penalty breakeven
    oracle_thresh = curtailment_penalty / (1 - critical_pct)
    oracle_mask = bt['RT_price'] > oracle_thresh
    oracle_cost = (bt['RT_price'] * capacity_mw).copy()
    oracle_cost[oracle_mask] = (bt.loc[oracle_mask, 'RT_price'] * capacity_mw * critical_pct) + (curtailment_penalty * flex_mw)
    oracle_savings = cost_always_on - oracle_cost.sum()

    def calc(curtail_mask, name):
        cost = (bt['RT_price'] * capacity_mw).copy()
        cost[curtail_mask] = (bt.loc[curtail_mask, 'RT_price'] * load_half_shed) + (curtailment_penalty * flex_mw * 0.5)
        savings = cost_always_on - cost.sum()
        pct = savings / oracle_savings * 100 if oracle_savings > 0 else 0
        return {'strategy': name, 'savings': savings,
                'hours_curtailed': curtail_mask.sum(), 'pct_oracle': pct}

    strategies = []
    strategies.append({'strategy': 'Always-on', 'savings': 0,
                       'hours_curtailed': 0, 'pct_oracle': 0})

    for t in [50, 60, 80]:
        strategies.append(calc(bt['DAM_price'] > t, f'DAM-only @${t}'))

    for t in [50, 60, 80]:
        lag_react = bt['RT_price'].shift(1).fillna(0) > t
        strategies.append(calc(lag_react, f'RT lag-react @${t}'))

    strategies.append(calc(bt['pred_prc'] < prc_tight, 'PRC model'))

    for t in [price_threshold]:
        strategies.append(calc(bt['pred_price'] > t, f'Price model @${t}'))
        combined = (bt['pred_prc'] < prc_tight) | (bt['pred_price'] > t)
        strategies.append(calc(combined, f'Combined system @${t}'))

    strategies.append({'strategy': f'Oracle @>${oracle_thresh:.0f}',
                       'savings': oracle_savings,
                       'hours_curtailed': oracle_mask.sum(), 'pct_oracle': 100})

    results = pd.DataFrame(strategies)

    if verbose:
        print(f"\n{'Strategy':<25} {'Savings':>12} {'Hours':>7} {'% Oracle':>9}")
        print('-' * 55)
        for _, r in results.iterrows():
            print(f"{r['strategy']:<25} ${r['savings']:>10,.0f} {r['hours_curtailed']:>7,} "
                  f"{r['pct_oracle']:>8.1f}%")

    return results


def compare_strategies_4cp(test_df, p_4cp, intervals_df,
                           capacity_mw=200, mining_revenue_per_mwh=40,
                           transmission_rate=45000, verbose=True):
    """Compare 4CP avoidance strategies.

    Unlike curtailment backtests (which compare savings vs baselines), 4CP compares
    PRECISION: everyone tries to dodge all peaks — the question is how many hours
    you curtail to do it. Fewer hours = less lost production = more efficient.

    Strategies:
    - No management: run during all peaks, pay full transmission
    - Conservative: curtail every summer afternoon 2-7 PM (what cautious operators do)
    - Our model at various thresholds: curtail only when P(4CP) > threshold
    - Oracle: curtail only the actual peak hours
    """
    bt = test_df[['timestamp', 'RT_price', 'month', 'hour']].copy()
    bt['year'] = bt['timestamp'].dt.year
    bt['p_4cp'] = p_4cp

    # Label actual peaks
    intervals_df = intervals_df.copy()
    intervals_df['timestamp'] = pd.to_datetime(intervals_df['timestamp'])
    bt['is_peak'] = bt['timestamp'].isin(intervals_df['timestamp']).astype(int)

    # Only summer hours
    summer = bt[bt['month'].isin([6, 7, 8, 9])].copy()
    years = sorted(summer['year'].unique())

    full_transmission = capacity_mw * transmission_rate  # annual bill if on during all peaks

    if verbose:
        print(f"\n{'='*75}")
        print(f"4CP TRANSMISSION AVOIDANCE (200 MW, ${transmission_rate:,}/MW/yr)")
        print(f"{'='*75}")
        print(f"Full transmission bill if on during all peaks: ${full_transmission:,.0f}/year")

    strategies = []

    # 1. No management — run full load all summer
    strategies.append({
        'strategy': 'No 4CP management',
        'hours_curtailed_per_summer': 0,
        'peaks_dodged': 0,
        'peaks_total': len(intervals_df[intervals_df['year'].isin(years)]),
        'transmission_per_year': full_transmission,
        'production_lost_per_summer': 0,
        'net_cost_per_year': full_transmission,
    })

    # 2. Conservative — curtail every summer afternoon (2-7 PM weekdays)
    afternoon_mask = (summer['hour'].between(14, 19))
    afternoons_per_year = {}
    for yr in years:
        yr_mask = (summer['year'] == yr) & afternoon_mask
        afternoons_per_year[yr] = yr_mask.sum()
    avg_afternoons = sum(afternoons_per_year.values()) / len(years)

    # Conservative catches all peaks (they always fall in this window)
    strategies.append({
        'strategy': 'Conservative (all afternoons)',
        'hours_curtailed_per_summer': int(avg_afternoons),
        'peaks_dodged': 4,
        'peaks_total': 4,
        'transmission_per_year': 0,
        'production_lost_per_summer': int(avg_afternoons) * mining_revenue_per_mwh * capacity_mw,
        'net_cost_per_year': int(avg_afternoons) * mining_revenue_per_mwh * capacity_mw,
    })

    # 3. Our model at various thresholds
    for thresh in [0.01, 0.05, 0.10, 0.20, 0.50]:
        total_curtailed = 0
        total_peaks_dodged = 0
        total_peaks = 0

        for yr in years:
            yr_summer = summer[summer['year'] == yr]
            yr_curtail = yr_summer['p_4cp'] > thresh
            yr_peaks = yr_summer['is_peak']

            total_curtailed += yr_curtail.sum()
            total_peaks += yr_peaks.sum()
            total_peaks_dodged += (yr_curtail & (yr_peaks == 1)).sum()

        n_years = len(years)
        avg_curtailed = total_curtailed / n_years
        avg_peaks_missed = (total_peaks - total_peaks_dodged) / n_years

        # Transmission: based on average peaks missed across years
        avg_4cp_demand = (avg_peaks_missed / 4) * capacity_mw
        annual_transmission = avg_4cp_demand * transmission_rate
        annual_production_lost = avg_curtailed * mining_revenue_per_mwh * capacity_mw

        strategies.append({
            'strategy': f'Model @{thresh:.2f}',
            'hours_curtailed_per_summer': int(avg_curtailed),
            'peaks_dodged': int(total_peaks_dodged),
            'peaks_total': int(total_peaks),
            'transmission_per_year': annual_transmission,
            'production_lost_per_summer': annual_production_lost,
            'net_cost_per_year': annual_transmission + annual_production_lost,
        })

    # 4. Oracle — curtail only actual peak hours
    peaks_per_year = total_peaks / len(years)
    strategies.append({
        'strategy': 'Oracle (peak hours only)',
        'hours_curtailed_per_summer': int(peaks_per_year),
        'peaks_dodged': int(total_peaks),
        'peaks_total': int(total_peaks),
        'transmission_per_year': 0,
        'production_lost_per_summer': int(peaks_per_year) * mining_revenue_per_mwh * capacity_mw,
        'net_cost_per_year': int(peaks_per_year) * mining_revenue_per_mwh * capacity_mw,
    })

    results = pd.DataFrame(strategies)

    if verbose:
        print(f"\n{'Strategy':<30} {'Hrs/summer':>11} {'Peaks':>7} {'Transmission':>13} "
              f"{'Lost prod':>11} {'Net cost/yr':>12}")
        print('-' * 90)
        for _, r in results.iterrows():
            peaks_str = f"{r['peaks_dodged']}/{r['peaks_total']}"
            print(f"{r['strategy']:<30} {r['hours_curtailed_per_summer']:>10,} {peaks_str:>7} "
                  f"${r['transmission_per_year']:>11,.0f} "
                  f"${r['production_lost_per_summer']:>9,.0f} "
                  f"${r['net_cost_per_year']:>10,.0f}")

        # Highlight the value
        no_mgmt = results.iloc[0]['net_cost_per_year']
        conservative = results.iloc[1]['net_cost_per_year']
        best_model = results[results['strategy'].str.startswith('Model')].sort_values('net_cost_per_year').iloc[0]
        print(f"\n--- Value of precision ---")
        print(f"  vs no management:  save ${no_mgmt - best_model['net_cost_per_year']:>10,.0f}/year")
        print(f"  vs conservative:   save ${conservative - best_model['net_cost_per_year']:>10,.0f}/year "
              f"({int(results.iloc[1]['hours_curtailed_per_summer'] - best_model['hours_curtailed_per_summer'])} fewer hours curtailed)")

    return results


def optimize_price_threshold_datacenter(val_df, predicted_price, capacity_mw=200,
                                        critical_pct=0.65, curtailment_penalty=50,
                                        verbose=True):
    """Find optimal price curtailment threshold for datacenter.

    Datacenter sheds half of flexible load (35% * 0.5 = 17.5% of total) when
    predicted price exceeds threshold. The trade-off: reduced electricity cost
    vs curtailment penalty ($50/MWh on the shed capacity).

    Worth curtailing when: RT_price * shed_MW > penalty * shed_MW
    i.e. when RT_price > penalty. But we're predicting, not observing, so
    false positives (curtail when RT was cheap) cost penalty with no savings.
    """
    flex_mw = capacity_mw * (1 - critical_pct)
    shed_mw = flex_mw * 0.5  # half of flexible load
    best_thresh = None
    best_net = -np.inf
    sweep = []

    for thresh in np.arange(30, 125, 5):
        curtail = predicted_price > thresh
        actual_rt = val_df['RT_price'].values

        # Savings from shedding: (RT_price - penalty) * shed_MW for each curtailed hour
        # Positive when RT > penalty (saved money), negative when RT < penalty (paid penalty for nothing)
        hourly_benefit = np.where(curtail, (actual_rt - curtailment_penalty) * shed_mw, 0)
        net = hourly_benefit.sum()

        n_curtail = int(curtail.sum())
        n_beneficial = int((curtail & (actual_rt > curtailment_penalty)).sum())
        n_harmful = int((curtail & (actual_rt <= curtailment_penalty)).sum())

        sweep.append({
            'threshold': float(thresh),
            'hours_curtailed': n_curtail,
            'beneficial_hours': n_beneficial,
            'harmful_hours': n_harmful,
            'net_savings': float(net),
        })

        if net > best_net:
            best_net = net
            best_thresh = float(thresh)

    if verbose:
        print("\n--- Price threshold sweep (datacenter) ---")
        print(f"{'thresh':>7} {'curtail':>8} {'good':>5} {'bad':>5} {'net':>12}")
        for r in sweep:
            marker = " ◄" if r['threshold'] == best_thresh else ""
            print(f"  ${r['threshold']:>5.0f} {r['hours_curtailed']:>7} {r['beneficial_hours']:>5} "
                  f"{r['harmful_hours']:>5} ${r['net_savings']:>10,.0f}{marker}")
        print(f"\n  Best: ${best_thresh:.0f} → net savings ${best_net:,.0f}")

    return best_thresh, best_net, sweep

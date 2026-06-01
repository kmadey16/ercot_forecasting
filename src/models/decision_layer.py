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

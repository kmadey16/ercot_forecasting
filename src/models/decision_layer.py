import pandas as pd
import numpy as np


# Mining backtest

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


# datacenter backtest

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


# Optimize Thresholds
def optimize_thresholds(val_df, predicted_prc, backtest_fn, capacity_mw=200, verbose=False, **kwargs):
     best_savings = 0
     best_params = {}

     for s_threshold in range(2000, 4500, 250):
          for t_threshold in range(4000,7000,250):
               if t_threshold <= s_threshold:
                    continue
               results = backtest_fn(val_df, predicted_prc, capacity_mw=capacity_mw, scarcity_threshold=s_threshold, tight_threshold=t_threshold,verbose=verbose, **kwargs)
               savings = results['always_on_cost'].sum() - results['guided_cost'].sum()

               if savings > best_savings:
                    best_savings = savings
                    best_params = {'tight_threshold': t_threshold, 'scarcity_threshold': s_threshold}
     
     return best_params, best_savings

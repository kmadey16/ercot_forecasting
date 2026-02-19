import pandas as pd
import numpy as np

# Mining backtest

def mining_backtest(test_df, predictions, breakeven=50, capacity_mw = 200):
     #breakeven: $/MWh cost to operate
     #capacity_mw: miner size in MW

     results = test_df[['timestamp', 'RT_price']].copy()
     results['regime'] = predictions

     #Cost for always-on
     results['always_on_cost'] = results['RT_price'] * capacity_mw

     #Model guided suggestions curtailing during tight/scarcity
     results['guided_cost'] = results['RT_price'] * capacity_mw
     results.loc[results['regime'].isin(['Tight', 'Scarcity']), 'guided_cost'] = 0

     #savings
     total_always_on = results['always_on_cost'].sum()
     total_guided = results['guided_cost'].sum()
     savings = total_always_on - total_guided

     hours_curtailed = (results['regime'].isin(['Tight', 'Scarcity'])).sum()

     print(f'Always-on cost: ${total_always_on:,.0f}')
     print(f'Model-guided cost: ${total_guided:,.0f}')
     print(f'Savings: ${savings:,.0f}')
     print(f'Hours curtailed: {hours_curtailed} / {len(results)}')
    
     return results


# datacenter backtest

def datacenter_backtest(test_df, predictions, critical_pct=0.65, capacity_mw=200):
     #Critical_pct: percentage of critical load
     #capacity_mw:  datacenter size in MW

     results = test_df[['timestamp', 'RT_price']].copy()
     results['regime'] = predictions

     flexible_capacity_pct = 1 - critical_pct

     # Always full power
     results['always_on_cost'] = results['RT_price'] * capacity_mw

     # Model guided
     results['guided_load'] = capacity_mw  # default full power
     results.loc[results['regime'] == 'Tight', 'guided_load'] = capacity_mw * (critical_pct + flexible_capacity_pct * 0.5)  # cut half flexible
     results.loc[results['regime'] == 'Scarcity', 'guided_load'] = capacity_mw * critical_pct  # cut all flexible
     results.loc[results['regime'] == 'Low', 'guided_load'] = capacity_mw  # run everything + batch jobs

     results['guided_cost'] = results['RT_price'] * results['guided_load']

     total_always_on = results['always_on_cost'].sum()
     total_guided = results['guided_cost'].sum()

     savings = total_always_on - total_guided

     print(f'Always-on cost: ${total_always_on:,.0f}')
     print(f'Model-guided cost: ${total_guided:,.0f}')
     print(f'Savings: ${savings:,.0f}')
     print(f'Hours at reduced load: {(results["guided_load"] < capacity_mw).sum()} / {len(results)}')
    
     return results
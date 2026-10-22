"""RLM1 summary from docs/ml/results/rlm1_runs/*.json. usage: python rlm1_report.py -> docs/ml/results/rlm1_runs.csv, rlm1_configs.csv"""
import glob,json,numpy as np,pandas as pd

if __name__=='__main__':
    rows=[]
    for p in sorted(glob.glob('docs/ml/results/rlm1_runs/*.json')):
        j=json.load(open(p));t=j['test'];best=max(j['val'],key=lambda v:v['sharpe'])
        rows.append({'coin':j['sym'][:3],'variant':j['variant'],'cost_bp':j['cost_bp'],'seed':j['seed'],'val_best_steps':best['steps'],'val_sharpe':best['sharpe'],
                     'val_trades_per_day':best['trades_per_day'],'val_time_in_market':best['time_in_market'],
                     **{f'test_{k}':v for k,v in t.items() if k!='by_year_return_pct'},**{f'ret_{y}':v for y,v in t['by_year_return_pct'].items()}})
    R=pd.DataFrame(rows);R.to_csv('docs/ml/results/rlm1_runs.csv',index=False)
    g=R.groupby(['coin','variant','cost_bp'])
    S=g.agg(seeds=('seed','count'),val_sharpe=('val_sharpe','mean'),test_sharpe=('test_sharpe','mean'),test_alpha_sharpe=('test_alpha_sharpe','mean'),
            alpha_pos=('test_alpha_sharpe',lambda q:int((q>0).sum())),beta=('test_beta','mean'),cagr_pct=('test_cagr_pct','mean'),max_dd_pct=('test_max_dd_pct','mean'),
            trades_per_day=('test_trades_per_day','mean'),time_in_market=('test_time_in_market','mean'),median_hold_min=('test_median_hold_min','median')).round(2)
    S['pass']=(S.test_alpha_sharpe>0.5)&(S.alpha_pos>=2)&(S.test_sharpe>0);S.to_csv('docs/ml/results/rlm1_configs.csv')
    pd.set_option('display.width',300);pd.set_option('display.max_columns',40);print(S.to_string());print(R.drop(columns=[c for c in R if c.startswith('ret_')]).to_string(index=False))

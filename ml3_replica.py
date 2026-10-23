"""Independent replication and stress test of crypto ML3 (weekly cross-sectional L/S in the top-N perps).
Own weekly accounting (buy-and-hold within the week, simple returns, delisted coins exit at their last close), own model
settings, rule-based factors to see what drives the result, and adversarial variants.
usage: python ml3_replica.py <panel dir with panel_close/qvol/fund.parquet>"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ml3")
SEL_END, VER_START = "2023-12-31", "2024-01-01"

def load():
    C = pd.read_parquet(D / "panel_close.parquet").astype(float)
    Q = pd.read_parquet(D / "panel_qvol.parquet").astype(float).reindex(C.index)
    F = pd.read_parquet(D / "panel_fund.parquet").astype(float).reindex(C.index)
    for x in (C, Q, F):
        x.index = pd.to_datetime(x.index).tz_localize(None)
    return C, Q, F

def universe(C, Q, F, n, min_age=90):
    live = F.notna().cumsum() > 0
    age = C.notna().cumsum()
    ok = live & (age >= min_age) & C.notna()
    tv = Q.rolling(30, min_periods=20).mean().where(ok)
    return tv.rank(axis=1, ascending=False) <= n, age

def weekly_panel(C, Q, F, n, delay=0):
    U, age = universe(C, Q, F, n)
    dec = C.index[C.index.dayofweek == 6]
    dec = dec[dec >= "2019-06-01"]
    Cf = C.ffill(limit=6)
    lc = np.log(Cf)
    R = np.log(C).diff()
    feats = {"mom1": lc - lc.shift(7), "mom4": lc - lc.shift(28), "mom12": lc - lc.shift(84),
             "vol4": R.rolling(28, min_periods=20).std(),
             "fund1": F.fillna(0).rolling(7).sum(), "fund4": F.fillna(0).rolling(28).sum(),
             "dvol": np.log(Q.rolling(7).sum() + 1) - np.log(Q.rolling(28).sum() / 4 + 1),
             "size": np.log(Q.rolling(30, min_periods=20).mean() + 1),
             "age": np.log(age.astype(float) + 1)}
    W = {k: v.loc[dec] for k, v in feats.items()}
    Ud = U.loc[dec]
    for k in list(W):
        W[f"{k}_rk"] = W[k].where(Ud).rank(axis=1, pct=True)
    entry = Cf.shift(-delay).loc[dec]
    exit_ = Cf.shift(-delay - 7).reindex(dec)
    fwd = exit_ / entry - 1
    fund_next = F.fillna(0).rolling(7).sum().shift(-delay - 7).reindex(dec)
    st = Ud.stack()
    idx = st[st].index
    rows = pd.DataFrame({k: v.stack(future_stack=True).reindex(idx) for k, v in W.items()})
    rows["fwd"] = fwd.stack(future_stack=True).reindex(idx)
    rows["fund_next"] = fund_next.stack(future_stack=True).reindex(idx)
    rows["y"] = rows.fwd - rows.groupby(level=0).fwd.transform("mean")
    return rows

def fit_predict(rows, feats, params, first=2021):
    pred = pd.Series(np.nan, index=rows.index)
    t = rows.index.get_level_values(0)
    for Y in range(first, t.max().year + 1):
        tr = (t + pd.Timedelta(days=8) < pd.Timestamp(f"{Y}-01-01")) & rows.y.notna() & (t >= "2020-01-01")
        te = t.year == Y
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(**params).fit(rows.loc[tr, feats], rows.loc[tr, "y"])
        pred[te] = m.predict(rows.loc[te, feats])
    return pred

def book_returns(rows, score, k=5, cost_bp=10.0, short_cost_bp=None, ls=True):
    df = rows[["fwd", "fund_next"]].assign(s=score).dropna(subset=["s", "fwd"])
    out, prev = [], pd.Series(dtype=float)
    for d, g in df.groupby(level=0):
        g = g.droplevel(0).sort_values("s")
        if len(g) < 2 * k:
            continue
        w = pd.Series(0.0, index=g.index)
        w[g.index[-k:]] = 0.5 / k if ls else 1.0 / k
        if ls:
            w[g.index[:k]] = -0.5 / k
        ret = (w * g.fwd).sum() - (w * g.fund_next).sum()
        turn = w.subtract(prev, fill_value=0).abs()
        sc = short_cost_bp if short_cost_bp is not None else cost_bp
        cost = (turn * np.where(w.reindex(turn.index).fillna(0) < 0, sc, cost_bp) * 1e-4).sum()
        out.append((d, ret - cost))
        prev = w * (1 + g.fwd)
    return pd.Series(dict(out)).sort_index()

def stats(r):
    if len(r) < 10:
        return {"sharpe": np.nan}
    eq = np.log1p(r.clip(lower=-0.99)).cumsum()
    return {"sharpe": round(r.mean() / r.std() * np.sqrt(52), 2), "cagr_pct": round((np.exp(eq.iloc[-1] * 52 / len(r)) - 1) * 100, 1),
            "max_dd_pct": round((1 - np.exp(-(eq.cummax() - eq).max())) * 100, 1), "weeks": len(r)}

def seg(r):
    return {"2021-23": stats(r[:SEL_END]), "2024-26": stats(r[VER_START:]),
            "by_year": {int(y): round(q.mean() / q.std() * np.sqrt(52), 2) for y, q in r.groupby(r.index.year) if len(q) > 8}}

def main():
    C, Q, F = load()
    base = ["mom1", "mom4", "mom12", "vol4", "fund1", "fund4", "dvol", "size"]
    feats = base + [f"{f}_rk" for f in base]
    theirs = dict(n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.03, min_child_samples=100, subsample=0.8, subsample_freq=1, verbose=-1)
    mine = dict(n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.05, min_child_samples=200, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=7)
    res = {}
    for n in (30, 50):
        rows = weekly_panel(C, Q, F, n)
        p2 = fit_predict(rows, feats, mine)
        res[f"N{n} ML their params"] = seg(book_returns(rows, fit_predict(rows, feats, theirs)))
        res[f"N{n} ML my params"] = seg(book_returns(rows, p2))
        res[f"N{n} ML my params + age"] = seg(book_returns(rows, fit_predict(rows, feats + ["age", "age_rk"], mine)))
        res[f"N{n} ML my, costs 30bp shorts"] = seg(book_returns(rows, p2, short_cost_bp=30))
        res[f"N{n} rule: size"] = seg(book_returns(rows, rows.size_rk))
        res[f"N{n} rule: age"] = seg(book_returns(rows, rows.age_rk))
        res[f"N{n} rule: size+age"] = seg(book_returns(rows, rows.size_rk + rows.age_rk))
        res[f"N{n} rule: -fund1"] = seg(book_returns(rows, -rows.fund1_rk))
        res[f"N{n} rule: mom4"] = seg(book_returns(rows, rows.mom4_rk))
        res[f"N{n} rule: -mom1"] = seg(book_returns(rows, -rows.mom1_rk))
        rd = weekly_panel(C, Q, F, n, delay=1)
        res[f"N{n} ML my, entry +1 day"] = seg(book_returns(rd, fit_predict(rd, feats, mine)))
        old = rows[np.exp(rows.age) > 365]
        res[f"N{n} ML my, coins >1y only"] = seg(book_returns(old, fit_predict(old, feats, mine)))
        calm = rows[rows.vol4_rk < 0.8]
        res[f"N{n} ML my, without top-20% vol"] = seg(book_returns(calm, fit_predict(calm, feats, mine)))
        print(f"N={n} done", flush=True)
    T = pd.DataFrame({k: {"sh_21_23": v["2021-23"]["sharpe"], "sh_24_26": v["2024-26"]["sharpe"], "dd_24_26": v["2024-26"].get("max_dd_pct"),
                          "cagr_24_26": v["2024-26"].get("cagr_pct"), "by_year": v["by_year"]} for k, v in res.items()}).T
    pd.set_option("display.width", 250)
    print(T.to_string())
    T.to_csv(Path(__file__).with_name("ml3_replica_results.csv"))

if __name__ == "__main__":
    main()

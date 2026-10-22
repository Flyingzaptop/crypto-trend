# ML3 panel (data only)

Built by `panel.py` on branch `claude/drawdowns-ml`.
- **Prices and turnover:** Binance spot daily klines for all 474 USDT pairs that ever had a USD-M perp, delisted pairs included (data.binance.vision monthly archives). The panel excludes stablecoins, gold and leveraged tokens, and starts 2019-01-01.
- **Funding:** Binance USD-M funding, summed per UTC day. For 1000X perps the funding of the `1000X` contract is used.

| file | content |
|---|---|
| `panel_close.parquet` | daily close (UTC day, close at 00:00 of the next day), float32 |
| `panel_qvol.parquet` | daily quote volume, USDT |
| `panel_fund.parquet` | daily funding sum. NaN before the first funding record, i.e. before the perp is listed; ML3 uses this as "perp is live" |
| `perps.csv` | spot→perp symbol map and first/last spot day, first/last funding day. If `perp_last_funding` < `spot_last`, the perp was delisted before spot |

ML3 universe on each Sunday: top-50 by 30-day mean turnover among coins that have a live perp (any funding seen up to t) and ≥ 90 days of spot history (`panel.universe`).

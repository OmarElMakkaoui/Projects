"""Synthetic dataset with the schema of the real one and a *known* answer.

The real Airparif extraction is not redistributable, so the pipeline is
validated on data where the truth is known:

  * a linear decline of 1.0 µg/m³ per year in the emission-driven baseline,
  * multiplicative lockdown effects of -40 %, -20 % and -15 %,
  * a meteorology that has no trend itself but drives large day-to-day swings
    (wind dilution, temperature inversions, rain washout, source direction),
  * weekday traffic effects and AR(1) day-to-day noise.

tests/test_pipeline.py checks that the pipeline recovers these numbers.
None of these values are claims about Paris.

    python make_synthetic_data.py --out data/synthetic_no2_paris.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TRUE_SLOPE = -1.0                     # µg/m³ per year, emission baseline
TRUE_LOCKDOWN = {"Lockdown 1": -0.40, "Lockdown 2": -0.20, "Lockdown 3": -0.15}


def make(seed: int = 0, start="1995-01-01", end="2023-12-31") -> pd.DataFrame:
    from airquality import LOCKDOWNS

    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    doy = days.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)

    def ar1(phi, sd):
        e = rng.normal(0, sd, n)
        out = np.empty(n)
        out[0] = e[0]
        for i in range(1, n):
            out[i] = phi * out[i - 1] + e[i]
        return out

    TM = 12.5 + 8.0 * season + ar1(0.8, 1.8)
    FFM = np.clip(3.2 - 0.6 * season + ar1(0.6, 0.9), 0.3, None)
    wet = rng.random(n) < 0.45
    RR = np.where(wet, rng.gamma(0.8, 4.5, n), 0.0).round(1)
    # prevailing south-westerlies, occasional north-easterly (continental) flow
    sw = rng.random(n) < 0.7
    DXY = np.where(sw, rng.vonmises(np.radians(225 - 180), 2.0, n),
                   rng.vonmises(np.radians(45 - 180), 2.5, n))
    DXY = (np.degrees(DXY) + 180) % 360

    years = (days - pd.Timestamp("1995-01-01")).days.to_numpy() / 365.25
    baseline = 55.0 + TRUE_SLOPE * years                       # emissions
    met = (-0.10 * (FFM - 3.2) - 0.012 * (TM - 12.5) - 0.02 * np.minimum(RR, 20)
           + 0.15 * np.cos(np.radians(DXY - 45)))              # log-scale effects
    weekday = days.dayofweek.to_numpy()
    traffic = np.select([weekday == 5, weekday == 6], [-0.12, -0.25], 0.0)
    lockdown = np.zeros(n)
    for name, (a, b) in LOCKDOWNS.items():
        lockdown[(days >= a) & (days <= b)] = np.log1p(TRUE_LOCKDOWN[name])

    multiplier = np.exp(met + traffic + ar1(0.5, 0.18))
    no2 = baseline * multiplier * np.exp(lockdown)

    df = pd.DataFrame({"AAAAMMJJ": days.strftime("%Y-%m-%d"), "CNO2": no2.round(1),
                       "RR": RR, "TM": TM.round(1), "FFM": FFM.round(1), "DXY": DXY.round(0)})
    # On the additive scale the analysis works on, the expected NO2 decline is
    # TRUE_SLOPE times the average non-emission multiplier (exp of met + traffic + noise).
    df.attrs["true_additive_slope"] = TRUE_SLOPE * float(multiplier[days <= "2019-12-31"].mean())
    # ~2 % missing values, as in the real extraction
    for c in ["CNO2", "FFM", "DXY"]:
        df.loc[rng.random(n) < 0.007, c] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_no2_paris.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    make(a.seed).to_csv(a.out, index=False)
    print(f"wrote {a.out}")

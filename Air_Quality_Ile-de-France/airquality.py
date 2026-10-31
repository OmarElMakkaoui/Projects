"""Meteorological normalisation of daily NO2 in Paris, trend estimation, and a
business-as-usual counterfactual for the COVID-19 lockdowns.

Data: one row per day with the columns of ``REQUIRED`` (see data/README.md).

Three analyses
--------------
1. Predictive skill of a meteorology-only GAM, evaluated with blocked
   leave-one-year-out cross-validation. Daily NO2 is strongly autocorrelated,
   so shuffled K-fold CV puts neighbouring days in train and test and
   overstates skill.

2. Long-term trend after removing meteorology. The GAM is fitted jointly with a
   smooth function of time,
       NO2_t = b0 + s_time(t) + f_TM + f_RR + f_FFM + f_DXY(cyclic) + f_doy(cyclic) + f_weekday + e_t,
   and the meteorologically adjusted series is NO2_t minus the four meteorological
   partial effects. Fitting time and meteorology together matters: residuals of
   a meteorology-only model would let any meteorological trend (warming, for
   instance) absorb part of the emission trend. The trend is then estimated on
   deseasonalised monthly means with the Theil-Sen slope (95% CI) and tested
   with the Mann-Kendall test.

3. COVID-19. A GAM trained on the three pre-pandemic years (meteorology,
   day of year, weekday; no time term, since splines do not extrapolate)
   predicts business-as-usual NO2 for 2020-2023. Observed minus predicted,
   averaged over each national lockdown, estimates the lockdown effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from scipy import stats

REQUIRED = ["AAAAMMJJ", "CNO2", "RR", "TM", "FFM", "DXY"]
MET = ["TM", "RR", "FFM", "DXY"]
MET_LABELS = {"TM": "mean temperature (°C)", "RR": "precipitation (mm)",
              "FFM": "mean wind speed (m/s)", "DXY": "wind direction (°)"}

# National lockdowns in France (start and end dates inclusive)
LOCKDOWNS = {
    "Lockdown 1": ("2020-03-17", "2020-05-10"),
    "Lockdown 2": ("2020-10-30", "2020-12-14"),
    "Lockdown 3": ("2021-04-03", "2021-05-02"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}; see data/README.md")
    df["date"] = pd.to_datetime(df["AAAAMMJJ"].astype(str), format="mixed", errors="raise")
    n0 = len(df)
    df = df.dropna(subset=REQUIRED[1:]).set_index("date").sort_index()
    if not df["DXY"].between(0, 360).all():
        raise ValueError("DXY (wind direction) must be in degrees, 0-360")
    df.attrs["dropped_fraction"] = 1 - len(df) / n0
    return add_calendar(df)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["doy"] = df.index.dayofyear
    df["weekday"] = df.index.dayofweek
    df["t"] = (df.index - pd.Timestamp("2000-01-01")).days / 365.25  # years since 2000
    return df


# ---------------------------------------------------------------------------
# GAM construction
# ---------------------------------------------------------------------------

def build_gam(features: list[str], n_splines: int = 20) -> LinearGAM:
    """Penalised regression splines; circular variables get cyclic splines."""
    terms = None
    for i, name in enumerate(features):
        if name == "DXY":
            term = s(i, basis="cp", edge_knots=[0, 360], n_splines=n_splines)
        elif name == "doy":
            term = s(i, basis="cp", edge_knots=[1, 367], n_splines=n_splines)
        elif name == "weekday":
            term = f(i)
        elif name == "t":
            term = s(i, n_splines=30)
        else:
            term = s(i, n_splines=n_splines)
        terms = term if terms is None else terms + term
    return LinearGAM(terms)


def fit_gam(df: pd.DataFrame, features: list[str],
            lam_grid=np.logspace(-1, 4, 6)) -> LinearGAM:
    """Fit by penalised least squares; the smoothing parameter is chosen by GCV
    over ``lam_grid`` (one value shared by all terms)."""
    gam = build_gam(features)
    gam.gridsearch(df[features].to_numpy(float), df["CNO2"].to_numpy(float),
                   lam=lam_grid, progress=False)
    return gam


def _scores(y, yhat) -> dict:
    resid = y - yhat
    return {"r2": 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "bias": float(resid.mean())}


# ---------------------------------------------------------------------------
# 1. Predictive skill
# ---------------------------------------------------------------------------

def leave_one_year_out(df: pd.DataFrame, features=MET, years=None) -> pd.DataFrame:
    """Blocked CV: each year is predicted by a model fitted on all other years."""
    years = sorted(set(df.index.year)) if years is None else years
    rows = []
    for yr in years:
        train, test = df[df.index.year != yr], df[df.index.year == yr]
        gam = fit_gam(train, features)
        pred = gam.predict(test[features].to_numpy(float))
        rows.append({"year": yr, "n_days": len(test), **_scores(test["CNO2"].to_numpy(), pred)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Meteorologically adjusted trend
# ---------------------------------------------------------------------------

TREND_FEATURES = ["t"] + MET + ["doy", "weekday"]


@dataclass
class TrendResult:
    gam: LinearGAM
    adjusted: pd.Series          # daily NO2 minus meteorological partial effects
    monthly: pd.Series           # deseasonalised monthly means of ``adjusted``
    slope_per_year: float        # Theil-Sen, µg/m3 per year
    slope_ci: tuple              # 95% CI
    mk_tau: float                # Mann-Kendall (Kendall's tau between time and series)
    mk_pvalue: float
    lag1_autocorr: float         # of the detrended monthly series


def meteorological_adjustment(df: pd.DataFrame) -> tuple[LinearGAM, pd.Series]:
    gam = fit_gam(df, TREND_FEATURES)
    X = df[TREND_FEATURES].to_numpy(float)
    met_effect = sum(gam.partial_dependence(term=TREND_FEATURES.index(v), X=X) for v in MET)
    return gam, pd.Series(df["CNO2"].to_numpy() - met_effect, index=df.index, name="adjusted")


def trend_test(series: pd.Series) -> dict:
    """Theil-Sen slope and Mann-Kendall test on deseasonalised monthly means."""
    monthly = series.resample("MS").mean().dropna()
    monthly = monthly - monthly.groupby(monthly.index.month).transform("mean") + monthly.mean()
    t = np.asarray((monthly.index - monthly.index[0]).days / 365.25)
    slope, intercept, lo, hi = stats.theilslopes(monthly.to_numpy(), t, alpha=0.95)
    tau, p = stats.kendalltau(t, monthly.to_numpy())
    detrended = monthly.to_numpy() - (intercept + slope * t)
    r1 = float(np.corrcoef(detrended[:-1], detrended[1:])[0, 1])
    return {"monthly": monthly, "slope_per_year": float(slope), "intercept": float(intercept),
            "slope_ci": (float(lo), float(hi)), "mk_tau": float(tau), "mk_pvalue": float(p),
            "lag1_autocorr": r1}


def adjusted_trend(df: pd.DataFrame) -> TrendResult:
    gam, adjusted = meteorological_adjustment(df)
    tt = trend_test(adjusted)
    return TrendResult(gam, adjusted, tt["monthly"], tt["slope_per_year"], tt["slope_ci"],
                       tt["mk_tau"], tt["mk_pvalue"], tt["lag1_autocorr"])


# ---------------------------------------------------------------------------
# 3. COVID-19 business-as-usual counterfactual
# ---------------------------------------------------------------------------

BAU_FEATURES = MET + ["doy", "weekday"]


def covid_counterfactual(df: pd.DataFrame, train=("2017-01-01", "2019-12-31"),
                         test=("2020-01-01", "2023-12-31")):
    """Returns (daily frame with observed / bau / ci, per-lockdown summary)."""
    tr, te = df.loc[train[0]:train[1]], df.loc[test[0]:test[1]]
    gam = fit_gam(tr, BAU_FEATURES)
    X = te[BAU_FEATURES].to_numpy(float)
    daily = pd.DataFrame({"observed": te["CNO2"], "bau": gam.predict(X)}, index=te.index)
    ci = gam.confidence_intervals(X, width=0.95)  # uncertainty of the mean prediction
    daily["bau_lo"], daily["bau_hi"] = ci[:, 0], ci[:, 1]
    daily["deviation"] = daily["observed"] - daily["bau"]

    # calibration of the BAU model: blocked CV inside the training window
    cv = leave_one_year_out(tr, BAU_FEATURES)

    rows = []
    for name, (a, b) in LOCKDOWNS.items():
        w = daily.loc[a:b]
        if len(w) == 0:
            continue
        rows.append({"period": name, "start": a, "end": b, "n_days": len(w),
                     "observed_mean": w["observed"].mean(), "bau_mean": w["bau"].mean(),
                     "deviation": w["deviation"].mean(),
                     "deviation_pct": 100 * w["deviation"].mean() / w["bau"].mean()})
    return daily, pd.DataFrame(rows), cv

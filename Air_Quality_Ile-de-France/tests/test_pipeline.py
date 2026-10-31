"""Validation on synthetic data with a known answer.  python -m pytest -q tests  (~1 min)"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import airquality as aq  # noqa: E402
import make_synthetic_data as syn  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = syn.make(seed=0)
    path = tmp_path_factory.mktemp("d") / "synthetic.csv"
    raw.to_csv(path, index=False)
    return aq.load_data(str(path)), raw.attrs["true_additive_slope"]


def test_schema_is_enforced(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"AAAAMMJJ": ["2020-01-01"], "CNO2": [30.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        aq.load_data(str(p))


def test_trend_recovered(data):
    df, true_slope = data
    tr = aq.adjusted_trend(df.loc[:"2019-12-31"])
    lo, hi = tr.slope_ci
    assert lo <= true_slope <= hi, (tr.slope_per_year, tr.slope_ci, true_slope)
    assert tr.mk_pvalue < 1e-6


def test_lockdown_effects_recovered(data):
    df, _ = data
    _, summary, _ = aq.covid_counterfactual(df)
    est = dict(zip(summary["period"], summary["deviation_pct"]))
    for name, truth in syn.TRUE_LOCKDOWN.items():
        assert abs(est[name] - 100 * truth) < 6, (name, est[name], truth)


def test_no_false_trend_on_trendless_series():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2000-01-01", "2019-12-31", freq="D")
    seasonal = 5 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    s = pd.Series(30 + seasonal + rng.normal(0, 6, len(idx)), index=idx)
    tt = aq.trend_test(s)
    assert tt["slope_ci"][0] < 0 < tt["slope_ci"][1]
    assert tt["mk_pvalue"] > 0.01


def test_wind_direction_effect_is_periodic(data):
    df, _ = data
    gam = aq.fit_gam(df.loc["2015":"2019"], aq.MET)
    i = aq.MET.index("DXY")
    X = np.tile(df[aq.MET].median().to_numpy(), (2, 1))
    X[:, i] = [0, 360]
    pd0, pd360 = gam.partial_dependence(term=i, X=X)
    assert np.isclose(pd0, pd360, atol=1e-6)


def test_blocked_cv_holds_out_whole_years(data):
    df, _ = data
    cv = aq.leave_one_year_out(df.loc["2016":"2019"], aq.MET)
    assert list(cv["year"]) == [2016, 2017, 2018, 2019]
    assert cv["n_days"].sum() == len(df.loc["2016":"2019"])

"""Run the three analyses and write tables and figures.

    python run_pipeline.py --data data/Q_Donnees.csv --out results/paris13
    python run_pipeline.py --data data/synthetic_no2_paris.csv --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import airquality as aq  # noqa: E402

warnings.filterwarnings("ignore", module="pygam")

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def plot_partial_effects(gam, df, path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, var in zip(axes, aq.MET):
        i = aq.TREND_FEATURES.index(var)
        XX = gam.generate_X_grid(term=i)
        pd_, ci = gam.partial_dependence(term=i, X=XX, width=0.95)
        x = XX[:, i]
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(x, pd_, color=BLUE, lw=2)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel(aq.MET_LABELS[var])
    axes[0].set_ylabel("partial effect on NO₂ (µg/m³)")
    fig.suptitle("Meteorological partial effects (GAM, 95% CI)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trend(tr, path):
    m = tr.monthly
    t = (m.index - m.index[0]).days / 365.25
    intercept = np.median(m.to_numpy() - tr.slope_per_year * t)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(m.index, m.to_numpy(), color=BLUE, lw=1.2, label="monthly mean, meteorology removed")
    ax.plot(m.index, intercept + tr.slope_per_year * t, color=ORANGE, lw=2, ls="--",
            label=f"Theil–Sen: {tr.slope_per_year:+.2f} µg/m³/yr "
                  f"[{tr.slope_ci[0]:+.2f}, {tr.slope_ci[1]:+.2f}]")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.set_title("Meteorologically adjusted NO₂ (deseasonalised monthly means)", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_covid(daily, path):
    d = daily.loc["2020-01-01":"2021-12-31"].rolling(7, center=True).mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, (a, b) in aq.LOCKDOWNS.items():
        ax.axvspan(np.datetime64(a), np.datetime64(b), color=GRID, alpha=0.8, lw=0)
        ax.text(np.datetime64(a), 0.98, f" {name}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=MUTED)
    ax.plot(d.index, d["bau"], color=ORANGE, lw=1.6, ls="--", label="business as usual (GAM, 2017–19)")
    ax.plot(d.index, d["observed"], color=BLUE, lw=1.6, label="observed")
    ax.set_ylabel("NO₂, 7-day mean (µg/m³)")
    ax.set_title("COVID-19: observed NO₂ vs weather-matched counterfactual", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cv", action="store_true", help="skip the 29-fold blocked CV")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = aq.load_data(a.data)
    print(f"{len(df)} days {df.index[0].date()} – {df.index[-1].date()}, "
          f"{100 * df.attrs['dropped_fraction']:.1f}% rows dropped for missing values")

    summary = {"n_days": len(df), "dropped_fraction": df.attrs["dropped_fraction"]}

    if not a.skip_cv:
        # Same blocked CV for two models: meteorology alone (as in the report) and
        # meteorology + smooth time trend + calendar. The gap is what the emission
        # trend, not the weather, contributes to out-of-year error.
        for name, feats in [("meteorology_only", aq.MET), ("meteorology_time_calendar", aq.TREND_FEATURES)]:
            cv = aq.leave_one_year_out(df.loc[:"2019-12-31"], feats)
            cv.to_csv(os.path.join(a.out, f"cv_{name}.csv"), index=False)
            summary[f"cv_{name}"] = {k: float(cv[k].mean()) for k in ["r2", "rmse", "mae"]}
            print(f"leave-one-year-out, {name} (1995–2019):",
                  {k: round(v, 3) for k, v in summary[f"cv_{name}"].items()})

    tr = aq.adjusted_trend(df.loc[:"2019-12-31"])
    summary["trend_1995_2019"] = {"slope_per_year": tr.slope_per_year, "ci95": tr.slope_ci,
                                  "mann_kendall_tau": tr.mk_tau, "mann_kendall_p": tr.mk_pvalue,
                                  "lag1_autocorr_monthly": tr.lag1_autocorr}
    print("trend:", summary["trend_1995_2019"])
    plot_partial_effects(tr.gam, df, os.path.join(a.out, "partial_effects.png"))
    plot_trend(tr, os.path.join(a.out, "adjusted_trend.png"))

    daily, lockdowns, bau_cv = aq.covid_counterfactual(df)
    lockdowns.to_csv(os.path.join(a.out, "covid_lockdowns.csv"), index=False)
    summary["bau_model_cv_2017_2019"] = {k: float(bau_cv[k].mean()) for k in ["r2", "rmse", "bias"]}
    summary["lockdowns"] = lockdowns.round(3).to_dict(orient="records")
    print(lockdowns.round(2).to_string(index=False))
    plot_covid(daily, os.path.join(a.out, "covid_counterfactual.png"))

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)


if __name__ == "__main__":
    main()

"""Figures for the README, from the CSVs in results/.

    python benchmarks/make_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# fixed categorical order, validated for colour-vision deficiency; markers as 2nd channel
STYLE = {
    "HC": ("#2a78d6", "o"), "GHC": ("#eb6834", "s"), "SA": ("#1baf7a", "^"),
    "TabuSA": ("#eda100", "D"), "BO": ("#e87ba4", "v"),
}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})
FLOOR = 1 / 1_252_440  # one configuration: plotted value when the optimum is found


def convergence():
    runs = pd.read_csv(os.path.join(RES, "methods_runs.csv"))
    cps = [int(c.split("@")[1]) for c in runs.columns if c.startswith("rq@")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for k, ax in enumerate(axes):
        for m, (color, marker) in STYLE.items():
            g = runs[(runs.sim == k) & (runs.method == m)]
            med = [max(np.median(g[f"rq@{c}"]), FLOOR) for c in cps]
            ax.plot(cps, med, color=color, marker=marker, ms=6, lw=2, label=m)
        ax.set_yscale("log")
        ax.set_title(f"Simulator {k}", loc="left")
        ax.set_xlabel("simulator evaluations")
        ax.axhline(FLOOR, color=MUTED, lw=0.8, ls=":")
    axes[0].set_ylabel("median fraction of configurations\nbetter than incumbent (log)")
    axes[0].text(cps[0], FLOOR * 1.4, "global optimum found", ha="left", va="bottom",
                 fontsize=8, color=MUTED)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Single-start search quality vs evaluation budget (30 starts each)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "convergence.png"), dpi=150)
    plt.close(fig)


def multistart():
    df = pd.read_csv(os.path.join(RES, "multistart.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True, sharey=True)
    for r, regime in enumerate(["weak", "strong"]):
        for k in range(3):
            ax = axes[r, k]
            for m in ["HC", "GHC", "SA", "TabuSA"]:
                g = df[(df.regime == regime) & (df.sim == k) & (df.method == m)].sort_values("P")
                color, marker = STYLE[m]
                ax.plot(g["P"], g["hit_rate"], color=color, marker=marker, ms=6, lw=2, label=m)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
            ax.set_ylim(-0.03, 1.03)
            if r == 0:
                ax.set_title(f"Simulator {k}", loc="left")
            if r == 1:
                ax.set_xlabel("MPI processes P")
        axes[r, 0].set_ylabel(("300 evals per process" if regime == "weak"
                               else "4,800 evals in total") + "\nP(global optimum found)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Parallel multi-start: probability of returning the exact optimum (30 repetitions)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "multistart.png"), dpi=150)
    plt.close(fig)


def scaling():
    path = os.path.join(RES, "mpi_scaling.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(df.P, df.P, color=MUTED, lw=1, ls="--", label="ideal")
    ax.plot(df.P, df.speedup, color=STYLE["HC"][0], marker="o", ms=6, lw=2, label="measured")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(df.P)
    ax.set_yticks(df.P)
    ax.set_xlabel("MPI processes P")
    ax.set_ylabel("speedup T(1) / T(P)")
    ax.set_title("Strong scaling, 1,600 evaluations", loc="left")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(df.P, 100 * df.efficiency, color=STYLE["HC"][0], marker="o", ms=6, lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(df.P, [str(p) for p in df.P])
    ax.set_ylim(0, 105)
    ax.set_xlabel("MPI processes P")
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_title("Efficiency = speedup / P", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "mpi_scaling.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    convergence()
    multistart()
    scaling()

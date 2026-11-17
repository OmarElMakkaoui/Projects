"""MLP vs GAM vs Random Forest for daily NO2, and Adam vs K-FAC vs SGD for the MLP.

Same task as the GAM in airquality.py: predict daily NO2 from time, weather,
day of year and weekday. Years are split into blocks so that no test day has
its neighbours in the training set:

    test        1999, 2004, 2009, 2014, 2019
    validation  1998, 2003, 2008, 2013, 2018
    train       every other year from 1995 to 2019

Every optimiser trains the same MLP (15 -> 64 -> 64 -> 1, ReLU) for the same
number of epochs with the same batch size. The learning rate and the stopping
epoch are both picked on the validation years (early stopping), then each
optimiser is rerun with 5 seeds and scored on the test years at the epoch
where its validation error was lowest. Training error is tracked every epoch
to compare how fast each optimiser actually fits.

    python nn_benchmark.py --data data/synthetic_no2_paris.csv --out results/synthetic_mlp
"""

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn

import airquality as aq
from kfac import KFAC

warnings.filterwarnings("ignore", module="pygam")

TEST_YEARS = [1999, 2004, 2009, 2014, 2019]
VAL_YEARS = [1998, 2003, 2008, 2013, 2018]

EPOCHS = 150
BATCH = 128
LR_GRID = {
    "SGD": [1e-3, 3e-3, 0.01, 0.03, 0.1],
    "Adam": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "K-FAC": [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3],
}
TUNE_SEEDS = [100, 101]
TEST_SEEDS = [0, 1, 2, 3, 4]


# data ------------------------------------------------------------------------

def split(df):
    y = df.index.year
    test = df[np.isin(y, TEST_YEARS)]
    val = df[np.isin(y, VAL_YEARS)]
    train = df[~np.isin(y, TEST_YEARS + VAL_YEARS)]
    return train, val, test


def features(df):
    """Inputs for the MLP and the forest. Angles become (sin, cos) pairs."""
    dxy = np.radians(df["DXY"])
    doy = 2 * np.pi * df["doy"] / 365.25
    X = pd.DataFrame({
        "t": df["t"], "TM": df["TM"], "RR": np.log1p(df["RR"]), "FFM": df["FFM"],
        "dxy_sin": np.sin(dxy), "dxy_cos": np.cos(dxy),
        "doy_sin": np.sin(doy), "doy_cos": np.cos(doy),
    }, index=df.index)
    for d in range(7):
        X[f"wd{d}"] = (df["weekday"] == d).astype(float)
    return X


def prepare(path):
    df = aq.load_data(path)
    train, val, test = split(df)
    Xtr, Xva, Xte = features(train), features(val), features(test)
    cont = ["t", "TM", "RR", "FFM"]
    mu, sd = Xtr[cont].mean(), Xtr[cont].std()
    for X in (Xtr, Xva, Xte):
        X[cont] = (X[cont] - mu) / sd
    y_mu, y_sd = train["CNO2"].mean(), train["CNO2"].std()
    return {
        "frames": (train, val, test),
        "X": [torch.tensor(X.to_numpy(), dtype=torch.float32) for X in (Xtr, Xva, Xte)],
        "y": [torch.tensor(((d["CNO2"] - y_mu) / y_sd).to_numpy(), dtype=torch.float32)[:, None]
              for d in (train, val, test)],
        "y_mu": y_mu, "y_sd": y_sd,
    }


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


# baselines -------------------------------------------------------------------

def run_gam(data):
    train, _, test = data["frames"]
    t0 = time.perf_counter()
    gam = aq.fit_gam(train, aq.TREND_FEATURES)
    fit_time = time.perf_counter() - t0
    pred = gam.predict(test[aq.TREND_FEATURES].to_numpy(float))
    return {"model": "GAM", "fit_time_s": fit_time,
            "test_rmse": rmse(test["CNO2"], pred), "test_r2": r2(test["CNO2"], pred)}


def run_forest(data):
    train, val, test = data["frames"]
    Xtr, Xva, Xte = (features(d).to_numpy() for d in (train, val, test))
    best = None
    for leaf in (1, 5, 20):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, n_jobs=1,
                                   random_state=0).fit(Xtr, train["CNO2"])
        score = rmse(val["CNO2"], rf.predict(Xva))
        if best is None or score < best[0]:
            best = (score, leaf)
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=best[1], n_jobs=1,
                               random_state=0).fit(Xtr, train["CNO2"])
    fit_time = time.perf_counter() - t0
    pred = rf.predict(Xte)
    return {"model": "Random Forest", "min_samples_leaf": best[1], "fit_time_s": fit_time,
            "test_rmse": rmse(test["CNO2"], pred), "test_r2": r2(test["CNO2"], pred)}


# MLP -------------------------------------------------------------------------

def make_mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 1))


def make_optimizer(name, model, lr):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "K-FAC":
        return KFAC(model, lr=lr, damping=1e-2, momentum=0.9, stat_every=10, inv_every=50)
    raise ValueError(name)


def _eval_rmse(model, data, i):
    with torch.no_grad():
        pred = model(data["X"][i]) * data["y_sd"] + data["y_mu"]
    return rmse(data["y"][i] * data["y_sd"] + data["y_mu"], pred)


def train_mlp(data, opt_name, lr, seed):
    """Train for EPOCHS epochs. Returns per-epoch RMSE on train, validation and
    test (in µg/m3) and the cumulative training time, which excludes evaluation."""
    torch.manual_seed(seed)
    Xtr, ytr = data["X"][0], data["y"][0]
    model = make_mlp(Xtr.shape[1])
    opt = make_optimizer(opt_name, model, lr)
    gen = torch.Generator().manual_seed(seed)

    curve, train_time, n_backward = [], 0.0, 0
    for _ in range(EPOCHS):
        t0 = time.perf_counter()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            out = model(Xtr[idx])
            if isinstance(opt, KFAC) and opt.needs_curvature():
                opt.update_curvature(out)
                n_backward += 1
            loss = 0.5 * ((out - ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            n_backward += 1
            opt.step()
        train_time += time.perf_counter() - t0
        tr, va, te = (_eval_rmse(model, data, i) for i in range(3))
        curve.append({"train": tr, "val": va, "test": te, "time_s": train_time})
        if not np.isfinite(va):
            break
    return {"curve": pd.DataFrame(curve), "train_time_s": train_time, "n_backward": n_backward}


# parallel driver -------------------------------------------------------------

_DATA = None


def _init(path):
    global _DATA
    torch.set_num_threads(1)
    _DATA = prepare(path)


def _job(args):
    opt_name, lr, seed = args
    return opt_name, lr, seed, train_mlp(_DATA, opt_name, lr, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    data = prepare(a.data)
    n = [len(f) for f in data["frames"]]
    print(f"train/val/test days: {n}")

    # learning rate chosen on the validation years, with early stopping
    tune_jobs = [(o, lr, s) for o, lrs in LR_GRID.items() for lr in lrs for s in TUNE_SEEDS]
    with Pool(os.cpu_count(), initializer=_init, initargs=(a.data,)) as pool:
        tuned = pool.map(_job, tune_jobs, chunksize=1)
    tuning = pd.DataFrame([{"optimizer": o, "lr": lr, "seed": s,
                            "best_val_rmse": r["curve"]["val"].min(),
                            "final_val_rmse": r["curve"]["val"].iloc[-1]}
                           for o, lr, s, r in tuned])
    tuning.to_csv(os.path.join(a.out, "lr_tuning.csv"), index=False)
    mean_val = tuning.groupby(["optimizer", "lr"])["best_val_rmse"].mean()
    best_lr = {o: float(mean_val[o].idxmin()) for o in LR_GRID}
    print("selected learning rates:", best_lr)

    final_jobs = [(o, best_lr[o], s) for o in LR_GRID for s in TEST_SEEDS]
    with Pool(os.cpu_count(), initializer=_init, initargs=(a.data,)) as pool:
        final = pool.map(_job, final_jobs, chunksize=1)

    rows, curves = [], []
    for o, lr, s, r in final:
        c = r["curve"]
        stop = int(c["val"].idxmin())          # early-stopping epoch (0-based)
        best_val = c["val"].iloc[stop]
        reach = int((c["val"] <= 1.02 * best_val).idxmax())
        rows.append({"model": f"MLP + {o}", "optimizer": o, "lr": lr, "seed": s,
                     "test_rmse": c["test"].iloc[stop],
                     "final_epoch_test_rmse": c["test"].iloc[-1],
                     "stop_epoch": stop + 1,
                     "epochs_to_2pct_val": reach + 1,
                     "seconds_to_2pct_val": c["time_s"].iloc[reach],
                     "train_rmse_epoch5": c["train"].iloc[4],
                     "train_rmse_epoch150": c["train"].iloc[-1],
                     "train_time_s": r["train_time_s"],
                     "time_per_epoch_ms": 1000 * r["train_time_s"] / len(c),
                     "backward_passes": r["n_backward"]})
        curves.append(c.assign(optimizer=o, seed=s, epoch=np.arange(1, len(c) + 1)))
    runs = pd.DataFrame(rows)
    y_test = data["frames"][2]["CNO2"].to_numpy()
    runs["test_r2"] = 1 - runs["test_rmse"] ** 2 / np.var(y_test)
    runs.to_csv(os.path.join(a.out, "mlp_runs.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(a.out, "mlp_curves.csv"), index=False)

    baselines = [run_gam(data), run_forest(data)]
    oracle = oracle_rmse(a.data, data["frames"][2])

    summary = (runs.groupby("model")
                   .agg(lr=("lr", "first"),
                        test_rmse=("test_rmse", "mean"), test_rmse_sd=("test_rmse", "std"),
                        test_r2=("test_r2", "mean"),
                        final_epoch_test_rmse=("final_epoch_test_rmse", "mean"),
                        stop_epoch=("stop_epoch", "median"),
                        epochs_to_2pct_val=("epochs_to_2pct_val", "median"),
                        seconds_to_2pct_val=("seconds_to_2pct_val", "median"),
                        train_rmse_epoch5=("train_rmse_epoch5", "mean"),
                        train_rmse_epoch150=("train_rmse_epoch150", "mean"),
                        train_time_s=("train_time_s", "mean"),
                        time_per_epoch_ms=("time_per_epoch_ms", "mean"),
                        backward_passes=("backward_passes", "mean"))
                   .reset_index())
    summary = pd.concat([pd.DataFrame(baselines).rename(columns={"fit_time_s": "train_time_s"}),
                         summary], ignore_index=True)
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"n_days": {"train": n[0], "val": n[1], "test": n[2]},
                   "selected_lr": best_lr, "oracle_test_rmse": oracle,
                   "epochs": EPOCHS, "batch": BATCH}, fh, indent=2)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    if oracle is not None:
        print(f"noise floor (true conditional mean) on test years: RMSE {oracle:.2f}")
    plot(a.out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = pd.read_csv(os.path.join(out, "mlp_curves.csv"))
    summary = pd.read_csv(os.path.join(out, "summary.csv")).set_index("model")
    with open(os.path.join(out, "summary.json")) as fh:
        oracle = json.load(fh)["oracle_test_rmse"]

    colors = {"SGD": "#2a78d6", "Adam": "#eb6834", "K-FAC": "#1baf7a"}
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "xtick.color": "#898781", "ytick.color": "#898781",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("train", "epoch", "Training RMSE per epoch"),
              ("test", "epoch", "Test RMSE per epoch"),
              ("val", "time_s", "Validation RMSE vs training time")]
    for ax, (col, x, title) in zip(axes, panels):
        for o, color in colors.items():
            g = curves[curves["optimizer"] == o].groupby(x if x == "epoch" else "epoch")
            mean = g[col].mean()
            xs = mean.index if x == "epoch" else g["time_s"].mean()
            lo, hi = g[col].min(), g[col].max()
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=color, lw=2, label=o)
        if oracle is not None:
            ax.axhline(oracle, color="#0b0b0b", lw=1, ls=":", label="noise floor")
        if col == "test":
            for name, ls in (("GAM", "--"), ("Random Forest", "-.")):
                ax.axhline(summary.loc[name, "test_rmse"], color="#898781", lw=1.2, ls=ls, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("epoch" if x == "epoch" else "training time (s)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("RMSE (µg/m³)")
    axes[1].set_ylim(8.0, 11.5)
    axes[2].set_ylim(8.3, 11.5)
    axes[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "optimizer_convergence.png"), dpi=150)
    plt.close(fig)


def oracle_rmse(path, test):
    """RMSE of the true conditional mean, available only for the synthetic data."""
    if "synthetic" not in os.path.basename(path):
        return None
    import make_synthetic_data as syn
    raw = syn.make()
    cm = pd.Series(raw.attrs["conditional_mean"], index=pd.to_datetime(raw["AAAAMMJJ"]))
    return rmse(test["CNO2"], cm.loc[test.index])


if __name__ == "__main__":
    main()

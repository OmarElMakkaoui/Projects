import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import nn_benchmark as nb  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return nb.prepare(os.path.join(ROOT, "data", "synthetic_no2_paris.csv"))


def test_year_blocks_do_not_overlap(data):
    years = [set(f.index.year) for f in data["frames"]]
    assert years[1] == set(nb.VAL_YEARS)
    assert years[2] == set(nb.TEST_YEARS)
    assert not (years[0] & years[1]) and not (years[0] & years[2])


def test_features_are_standardised_on_train_only(data):
    Xtr = data["X"][0].numpy()
    assert np.allclose(Xtr[:, :4].mean(0), 0, atol=1e-5)
    assert np.allclose(Xtr[:, :4].std(0), 1, atol=1e-2)


@pytest.mark.parametrize("opt", ["SGD", "Adam", "K-FAC"])
def test_short_training_run(data, monkeypatch, opt):
    monkeypatch.setattr(nb, "EPOCHS", 2)
    res = nb.train_mlp(data, opt, nb.LR_GRID[opt][2], seed=0)
    curve = res["curve"]
    assert len(curve) == 2
    assert np.isfinite(curve[["train", "val", "test"]].to_numpy()).all()
    # K-FAC does one extra backward pass on the steps where it refreshes curvature
    steps = 2 * int(np.ceil(len(data["X"][0]) / nb.BATCH))
    extra = (steps + 9) // 10 if opt == "K-FAC" else 0
    assert res["n_backward"] == steps + extra

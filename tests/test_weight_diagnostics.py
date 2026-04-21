import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export import predict_weights
from src.losses import cvar_smooth
from src.train import run_training


def _make_predictive_dataset(
    *,
    seed: int = 7,
    n_windows: int = 900,
    window_len: int = 20,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Build a deterministic synthetic dataset where the last observed signal predicts
    next-day IPO excess return.
    """
    rng = np.random.default_rng(seed)
    total = n_windows + window_len + 2

    state = np.zeros(total, dtype=np.float64)
    for t in range(1, total):
        state[t] = 0.87 * state[t - 1] + 0.55 * rng.standard_normal()

    signal = np.tanh(state)
    market = 0.00035 + 0.006 * rng.standard_normal(total)
    excess = np.zeros(total, dtype=np.float64)
    excess[1:] = 0.0028 * signal[:-1] + 0.0022 * rng.standard_normal(total - 1)
    ipo = market + excess

    feat = np.column_stack(
        [
            market,
            ipo,
            signal,
            np.r_[0.0, excess[:-1]],
            np.r_[0.0, market[:-1]],
        ]
    ).astype(np.float32)

    X, R, sig = [], [], []
    for t in range(window_len, window_len + n_windows):
        X.append(feat[t - window_len : t])
        R.append([market[t], ipo[t]])
        sig.append(signal[t - 1])
    X = np.asarray(X, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    sig = np.asarray(sig, dtype=np.float32)

    cut = int(0.7 * n_windows)
    data = {
        "X_train": X[:cut],
        "R_train": R[:cut],
        "dates_train": np.arange(cut),
        "X_val": X[cut:],
        "R_val": R[cut:],
        "dates_val": np.arange(n_windows - cut),
        "feature_cols": [],
        "df": None,
        "n_assets": 2,
        "window_len": window_len,
    }
    return data, sig[cut:], R[cut:]


def _fit_and_eval(
    data: dict,
    signal_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    seed: int,
    **loss_kw,
) -> dict[str, float]:
    torch.manual_seed(seed)
    model, _ = run_training(
        data,
        device=torch.device("cpu"),
        epochs=12,
        lr=2e-3,
        batch_size=64,
        patience=6,
        model_type="gru",
        hidden_size=32,
        verbose=False,
        log_every=0,
        weight_decay=1e-5,
        dropout=0.0,
        **loss_kw,
    )
    w = predict_weights(model, data["X_val"], torch.device("cpu"))
    ipo_w = w[:, 1]
    model_ret = (w * returns_val).sum(axis=1)
    eq_ret = 0.5 * returns_val[:, 0] + 0.5 * returns_val[:, 1]
    return {
        "ipo_std": float(np.std(ipo_w)),
        "corr_weight_signal": float(np.corrcoef(ipo_w, signal_val)[0, 1]),
        "mean_model_return": float(np.mean(model_ret)),
        "mean_equal_weight_return": float(np.mean(eq_ret)),
    }


def test_cvar_smooth_focuses_left_tail():
    x = torch.tensor([-0.20, -0.10, 0.01, 0.02, 0.03], dtype=torch.float32)
    cvar = float(cvar_smooth(x, alpha=0.4, temperature=0.05).item())
    worst_two_mean = float(torch.sort(x).values[:2].mean().item())
    overall_mean = float(x.mean().item())

    # Tail risk metric should stay close to downside outcomes, not the global mean.
    assert cvar < overall_mean
    assert cvar < -0.08
    assert abs(cvar - worst_two_mean) < 0.08

    x_worse = torch.tensor([-0.40, -0.10, 0.01, 0.02, 0.03], dtype=torch.float32)
    cvar_worse = float(cvar_smooth(x_worse, alpha=0.4, temperature=0.05).item())
    assert cvar_worse < cvar


def test_gru_learns_predictive_signal_and_moves_weights():
    data, signal_val, returns_val = _make_predictive_dataset(seed=11, n_windows=900)
    out = _fit_and_eval(
        data,
        signal_val,
        returns_val,
        seed=11,
        lambda_cvar=0.0,
        lambda_vol=0.0,
        lambda_turnover=0.0,
        lambda_path=0.0,
        lambda_diversify=0.0,
        lambda_vol_excess=0.0,
        mean_return_weight=1.0,
        log_growth_weight=0.0,
    )

    assert out["ipo_std"] > 0.20
    assert out["corr_weight_signal"] > 0.70
    assert out["mean_model_return"] > out["mean_equal_weight_return"] + 2e-4


def test_turnover_and_path_penalties_can_freeze_weights():
    data, signal_val, returns_val = _make_predictive_dataset(seed=7, n_windows=900)
    baseline = _fit_and_eval(
        data,
        signal_val,
        returns_val,
        seed=7,
        lambda_cvar=0.5,
        lambda_vol=0.5,
        lambda_turnover=0.01,
        lambda_path=0.01,
        lambda_diversify=0.0,
        lambda_vol_excess=1.0,
        target_vol_annual=0.25,
        mean_return_weight=1.0,
        log_growth_weight=0.0,
    )
    no_turn_path = _fit_and_eval(
        data,
        signal_val,
        returns_val,
        seed=7,
        lambda_cvar=0.5,
        lambda_vol=0.5,
        lambda_turnover=0.0,
        lambda_path=0.0,
        lambda_diversify=0.0,
        lambda_vol_excess=1.0,
        target_vol_annual=0.25,
        mean_return_weight=1.0,
        log_growth_weight=0.0,
    )

    assert baseline["ipo_std"] < 0.02
    assert no_turn_path["ipo_std"] > 0.20
    assert no_turn_path["ipo_std"] > baseline["ipo_std"] * 20.0

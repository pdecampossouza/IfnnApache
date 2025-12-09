"""
Script 4 – Interpretability of ENFSUni0Evolving on Apache 'due' dataset
=======================================================================

Goal
----
This script focuses on the *interpretability* aspects of ENFSUni0Evolving
on the Apache "due" dataset, in a data-stream scenario.

It reuses the same experimental configurations as Script 3, but only for
the ENFSUni0Evolving model, and concentrates on:

  * Tracking the **number of rules** over time (history_n_rules).
  * Tracking the **feature relevance weights** over time
    (history_feature_weights, separability-based).
  * Exporting a **textual fuzzy rule base** using generate_fuzzy_rules().

We consider four combinations:

  1) Feature sets:
       - fnn_selected   (features_fnn from Script 1)
       - all_features   (all numeric columns except identifiers/targets)

  2) Drift definitions (used only for visualization / marking drifts):
       - fixed3   (three drifts at 25%, 50%, 75% of stream)
       - real3    (three drifts detected from changes in delay rate)

For each (feature_mode, drift_mode), the script:

  - Loads X, y in temporal order (openeddate).
  - Standardizes features.
  - Runs ENFSUni0Evolving in a prequential fashion (predict→update).
  - Collects:
      * history_n_rules
      * history_feature_weights (via get_feature_weight_matrix())
  - Saves:
      * weights_time.npy     (T × d)
      * n_rules_time.npy     (T,)
      * rules_final.txt      (human-readable fuzzy rule base)
  - Plots:
      * weights_over_time.png
      * n_rules_over_time.png
        (both with drift points marked).

Required files (from Script 0 and Script 1):
    ../apachedataset/processed/apache_due_numeric.csv
    ../apachedataset/processed/feature_analysis/selected_features.json

Model import:
    from models.enfs_uni0_evolving import ENFSUni0Evolving
"""

import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# Non-interactive backend for figures
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler

# Make local models importable
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.enfs_uni0_evolving import ENFSUni0Evolving  # type: ignore


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_DIR = Path("../apachedataset/processed")
FEATURE_DIR = DATASET_DIR / "feature_analysis"

NUMERIC_CSV_NAME = "apache_due_numeric.csv"
SELECTED_FEATURES_JSON = "selected_features.json"

RESULTS_ROOT = FEATURE_DIR / "interpretability_enfsuni0_apache"

N_MAX = None  # optional cap on stream length (for quick tests)

FEATURE_MODES = ["fnn_selected", "all_features"]
DRIFT_MODES = ["fixed3", "real3"]


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------


def load_apache_numeric(feature_mode: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Apache 'due' numeric dataset and select features according to feature_mode.

    Parameters
    ----------
    feature_mode : {"fnn_selected", "all_features"}

    Returns
    -------
    X : np.ndarray, shape (N, d)
        Feature matrix in temporal order (sorted by openeddate).
    y : np.ndarray, shape (N,)
        Binary target (y_binary: 0 = no delay, 1 = delay).
    feature_names : list of str
        Names of the selected features.
    """
    csv_path = DATASET_DIR / NUMERIC_CSV_NAME
    sel_path = FEATURE_DIR / SELECTED_FEATURES_JSON

    if not csv_path.exists():
        raise FileNotFoundError(f"Numeric CSV not found: {csv_path}")
    if not sel_path.exists():
        raise FileNotFoundError(f"Selected features JSON not found: {sel_path}")

    df = pd.read_csv(csv_path)

    # Ensure temporal order
    if "openeddate" in df.columns:
        df["openeddate"] = pd.to_datetime(df["openeddate"])
        df = df.sort_values("openeddate").reset_index(drop=True)

    # Target
    if "y_binary" not in df.columns:
        raise ValueError("Column 'y_binary' not found in apache_due_numeric.csv")
    y = df["y_binary"].values.astype(int)

    # Feature selection
    if feature_mode == "fnn_selected":
        with open(sel_path, "r", encoding="utf-8") as f:
            feature_sets = json.load(f)
        feature_names = feature_sets["features_fnn"]

    elif feature_mode == "all_features":
        drop_cols = {"issuekey", "openeddate", "delaydays", "impact_class", "y_binary"}
        feature_names = [c for c in df.columns if c not in drop_cols]

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    X = df[feature_names].values.astype(float)

    # Optional: limit length for quick experiments
    if N_MAX is not None and X.shape[0] > N_MAX:
        X = X[:N_MAX]
        y = y[:N_MAX]

    return X, y, feature_names


# -------------------------------------------------------------------------
# Drift definitions (for visualization)
# -------------------------------------------------------------------------


def fixed_drifts(n_samples: int, n_drifts: int = 3) -> List[int]:
    """
    Return fixed drift points at equally spaced positions.

    Example (n_drifts=3):
        positions at 0.25 N, 0.50 N, 0.75 N
    """
    points = []
    for k in range(1, n_drifts + 1):
        pos = int(k * n_samples / (n_drifts + 1))
        if 0 < pos < n_samples:
            points.append(pos)
    return points


def detect_real_drifts_from_y(
    y: np.ndarray, n_drifts: int = 3, window: int = 300
) -> List[int]:
    """
    Detect drift points based on changes in delay rate (mean of y) over time.

    Strategy
    --------
    - Partition the stream into non-overlapping windows of size `window`.
    - Compute delay rate in each window.
    - Compute absolute differences between consecutive windows.
    - Select the `n_drifts` largest differences and place a drift point
      at the boundary between those windows.
    """
    n = len(y)
    if n < 2 * window:
        return fixed_drifts(n, n_drifts=n_drifts)

    n_windows = n // window
    means = np.array(
        [float(y[i * window : (i + 1) * window].mean()) for i in range(n_windows)]
    )

    diffs = np.abs(np.diff(means))  # length n_windows-1
    if diffs.size == 0:
        return fixed_drifts(n, n_drifts=n_drifts)

    idx_sorted = np.argsort(diffs)[::-1]
    idx_top = sorted(idx_sorted[: min(n_drifts, diffs.size)])

    drifts = []
    for idx in idx_top:
        pos = (idx + 1) * window
        if 0 < pos < n:
            drifts.append(pos)

    return sorted(set(drifts))


# -------------------------------------------------------------------------
# Plotting utilities
# -------------------------------------------------------------------------


def plot_weights_over_time(
    W: np.ndarray,
    feature_names: List[str],
    drift_points: List[int],
    outpath: Path,
    title_suffix: str = "",
) -> None:
    """
    Plot the evolution of feature weights over time.

    W : np.ndarray, shape (T, d)
        Each column j corresponds to a feature j.
    """
    if W.size == 0:
        return

    T, d = W.shape
    t = np.arange(T)

    plt.figure(figsize=(8, 5))
    for j in range(d):
        plt.plot(t, W[:, j], label=feature_names[j])

    for dp in drift_points:
        if 0 <= dp < T:
            plt.axvline(dp, color="red", linestyle="--", alpha=0.5)

    plt.xlabel("Samples")
    plt.ylabel("Feature weight")
    plt.ylim(0.0, 1.05)
    plt.title(f"ENFSUni0 – Feature weights over time {title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, loc="upper right", ncol=1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_rules_over_time(
    n_rules_time: np.ndarray,
    drift_points: List[int],
    outpath: Path,
    title_suffix: str = "",
) -> None:
    """
    Plot the evolution of the number of rules over time.
    """
    if n_rules_time.size == 0:
        return

    T = n_rules_time.size
    t = np.arange(T)

    plt.figure(figsize=(8, 4))
    plt.plot(t, n_rules_time, label="#rules")

    for dp in drift_points:
        if 0 <= dp < T:
            plt.axvline(dp, color="red", linestyle="--", alpha=0.5)

    plt.xlabel("Samples")
    plt.ylabel("#Rules")
    plt.title(f"ENFSUni0 – Rule evolution over time {title_suffix}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_name = "Apache_due"

    for feature_mode in FEATURE_MODES:
        print(f"\n=== Feature mode: {feature_mode} ===")
        X, y, feat_names = load_apache_numeric(feature_mode)
        n_samples, n_features = X.shape
        print(f"  -> {n_samples} samples, {n_features} features.")

        # Standardization
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Precompute drift points for visualization
        fixed3 = fixed_drifts(n_samples, n_drifts=3)
        real3 = detect_real_drifts_from_y(y, n_drifts=3, window=300)

        for drift_mode in DRIFT_MODES:
            print(f"  -- Drift mode: {drift_mode}")

            if drift_mode == "fixed3":
                drift_points = fixed3
            elif drift_mode == "real3":
                drift_points = real3
            else:
                raise ValueError(f"Unknown drift_mode: {drift_mode}")

            outdir = RESULTS_ROOT / feature_mode / drift_mode
            outdir.mkdir(parents=True, exist_ok=True)

            # Instantiate ENFSUni0Evolving with the same style as Script 3
            model = ENFSUni0Evolving(
                n_features=n_features,
                n_classes=2,
                max_rules=30,
                sim_threshold=0.95,
                buffer_size_separability=200,
            )

            # Prequential loop (only ENFSUni0; we still can track accuracy)
            correct = np.zeros(n_samples, dtype=float)
            acc = np.zeros(n_samples, dtype=float)

            for t in tqdm(
                range(n_samples),
                desc=f"ENFSUni0 interpretability ({feature_mode}, {drift_mode})",
                leave=False,
            ):
                x_t = Xs[t]
                y_t = int(y[t])

                # Predict before seeing the label
                if t == 0 or model.n_rules_ == 0:
                    y_hat = 0
                else:
                    y_hat = model.predict_one(x_t)

                correct[t] = 1.0 if y_hat == y_t else 0.0
                acc[t] = correct[: t + 1].mean()

                # Update model
                model.learn_one(x_t, y_t)

            print(
                f"[RESULT] feat={feature_mode} | drift={drift_mode} | "
                f"Acc_final={acc[-1]:.4f} | Acc_mean={acc.mean():.4f} | "
                f"Rules_final={model.n_rules_}"
            )

            # -----------------------------------------------------------------
            # Save interpretability artefacts
            # -----------------------------------------------------------------

            # 1) Trajectory of feature weights (T × d)
            W_traj = model.get_feature_weight_matrix()
            np.save(outdir / "weights_time.npy", W_traj)

            # 2) Trajetória de nº de regras (T,)
            n_rules_time = np.array(model.history_n_rules, dtype=float)
            np.save(outdir / "n_rules_time.npy", n_rules_time)

            # 3) Regras fuzzy finais (texto)
            rules_txt = model.generate_fuzzy_rules(feature_names=feat_names)
            with open(outdir / "rules_final.txt", "w", encoding="utf-8") as f:
                f.write(
                    f"# ENFSUni0 fuzzy rule base – {dataset_name}\n"
                    f"# features = {feature_mode}, drift_mode = {drift_mode}\n"
                    f"# final_accuracy = {acc[-1]:.4f}\n"
                    f"# mean_accuracy  = {acc.mean():.4f}\n"
                    f"# n_rules_final  = {model.n_rules_}\n\n"
                )
                for r in rules_txt:
                    f.write(r + "\n")

            # 4) Figuras – pesos ao longo do tempo
            if W_traj.size > 0:
                plot_weights_over_time(
                    W_traj,
                    feat_names,
                    drift_points,
                    outpath=outdir
                    / f"weights_over_time_{dataset_name}_{feature_mode}_{drift_mode}.png",
                    title_suffix=f"({dataset_name}, {feature_mode}, {drift_mode})",
                )

            # 5) Figuras – nº de regras ao longo do tempo
            if n_rules_time.size > 0:
                plot_rules_over_time(
                    n_rules_time,
                    drift_points,
                    outpath=outdir
                    / f"n_rules_over_time_{dataset_name}_{feature_mode}_{drift_mode}.png",
                    title_suffix=f"({dataset_name}, {feature_mode}, {drift_mode})",
                )


if __name__ == "__main__":
    main()

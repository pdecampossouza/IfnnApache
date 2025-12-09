"""
Script 3 – Stream experiments with evolving fuzzy models on Apache 'due' dataset
================================================================================

This script evaluates evolving fuzzy systems (ENFSUni0Evolving, ePL, ePL_plus,
exTS, Simpl_eTS, eMG, ePL_KRLS_DISCO) using the Apache “due” dataset in a
*streaming* scenario.

It runs four combinations:

1. Feature sets:
   - fnn_selected   (6 features chosen by Script 1)
   - all_features   (all numeric columns from Script 0)

2. Drift definitions:
   - fixed3   (three drifts at 25%, 50%, 75% of stream)
   - real3    (three drifts detected from changes in delay rate over windows)

For each combination it performs:

- Prequential evaluation (predict → update)
- Tracking accuracy over time
- Tracking number of rules over time
- Tracking:
    * Pred_fallbacks  (times we had to fall back to class 0 due to NaN/inf)
    * P_instabilities (for ENFSUni0: number of RLS “repairs”)
- Saving per-model plots
- Saving multi-model comparison plots
- Outputting a LaTeX table with accuracy, rules, and stability indicators

Requires Script 0 and Script 1 outputs:
    ../apachedataset/processed/apache_due_numeric.csv
    ../apachedataset/processed/feature_analysis/selected_features.json
"""

import os
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from pathlib import Path
import sys

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------------------
# Paths and imports
# -------------------------------------------------------------------------

# make models importable (we are in Pipeline/, models one level above)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# evolving fuzzy models
from models.enfs_uni0_evolving import ENFSUni0Evolving  # type: ignore
from evolvingfuzzysystems.eFS import (
    ePL,
    ePL_plus,
    exTS,
    Simpl_eTS,
    eMG,
    ePL_KRLS_DISCO,
)  # type: ignore

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_DIR = Path("../apachedataset/processed")
FEATURE_DIR = DATASET_DIR / "feature_analysis"

NUMERIC_CSV_NAME = "apache_due_numeric.csv"
SELECTED_FEATURES_JSON = "selected_features.json"

RESULTS_ROOT = FEATURE_DIR / "stream_experiments_apache"

# stream length cap (None = full length)
N_MAX = None

# warmup and batch sizes for Kaike eFS models
WARMUP = 200
BATCH_EVOLVE = 20

FEATURE_MODES = ["fnn_selected", "all_features"]
DRIFT_MODES = ["fixed3", "real3"]

MODEL_NAMES = [
    "ENFSUni0",
    "ePL",
    "ePL_plus",
    "exTS",
    "Simpl_eTS",
    "eMG",
    "ePL_KRLS_DISCO",
]


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------


def load_apache_numeric(feature_mode: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    csv_path = DATASET_DIR / NUMERIC_CSV_NAME
    sel_path = FEATURE_DIR / SELECTED_FEATURES_JSON

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not sel_path.exists():
        raise FileNotFoundError(sel_path)

    df = pd.read_csv(csv_path)

    # garantir ordem temporal
    if "openeddate" in df.columns:
        df["openeddate"] = pd.to_datetime(df["openeddate"])
        df = df.sort_values("openeddate").reset_index(drop=True)

    # target binário
    if "y_binary" not in df.columns:
        raise ValueError("Column 'y_binary' missing!")

    y = df["y_binary"].values.astype(int)

    if feature_mode == "fnn_selected":
        # usa exatamente o conjunto de features escolhido pelo Script 1
        with open(sel_path, "r", encoding="utf-8") as f:
            feature_sets = json.load(f)
        feature_names = feature_sets["features_fnn"]

    elif feature_mode == "all_features":
        # aqui vamos pegar SOMENTE colunas numéricas,
        # excluindo identificadores e alvos
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # colunas que NÃO queremos usar como preditores
        drop_cols = {"delaydays", "impact_class", "y_binary"}

        feature_names = [c for c in numeric_cols if c not in drop_cols]

    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    # monta X apenas com as colunas selecionadas
    X = df[feature_names].values.astype(float)

    # opcional: limitar o tamanho do stream
    if N_MAX is not None and X.shape[0] > N_MAX:
        X = X[:N_MAX]
        y = y[:N_MAX]

    return X, y, feature_names


# -------------------------------------------------------------------------
# Drift generation
# -------------------------------------------------------------------------


def fixed_drifts(n_samples: int, n_drifts: int = 3) -> List[int]:
    """Equally spaced drift points: 1/(n_drifts+1), 2/(n_drifts+1), ..."""
    d = []
    for k in range(1, n_drifts + 1):
        pos = int(k * n_samples / (n_drifts + 1))
        if 0 < pos < n_samples:
            d.append(pos)
    return d


def detect_real_drifts_from_y(
    y: np.ndarray, n_drifts: int = 3, window: int = 300
) -> List[int]:
    """
    Detect “real” drift points from changes in delay rate.

    Strategy:
      - split stream in windows of size `window`
      - compute mean(y) in each window
      - take |delta mean| between consecutive windows
      - choose boundaries with largest deltas as drift points
    """
    n = len(y)
    if n < 2 * window:
        return fixed_drifts(n, n_drifts)

    n_windows = n // window
    means = np.array(
        [float(y[i * window : (i + 1) * window].mean()) for i in range(n_windows)]
    )

    diffs = np.abs(np.diff(means))
    idx_sorted = np.argsort(diffs)[::-1]
    idx_top = sorted(idx_sorted[: min(n_drifts, len(diffs))])

    drifts = []
    for idx in idx_top:
        pos = (idx + 1) * window
        if 0 < pos < n:
            drifts.append(pos)

    return sorted(set(drifts))


# -------------------------------------------------------------------------
# SAFE WRAPPER FOR Kaike eFS MODELS (exTS, Simpl_eTS, eMG, ePL_plus)
# -------------------------------------------------------------------------


class SafeEFSWrapper:
    """
    Wraps an evolvingfuzzysystems model so that:
    - internal rule index errors do NOT crash the experiment
    - non-finite predictions fall back to class 0
    - keeps counters:
        * stabilizations (internal errors caught)
        * fallback_preds (NaN/inf predictions)

    This is especially useful for:
        exTS, Simpl_eTS, eMG, ePL_plus
    """

    def __init__(self, model, name="eFS"):
        self.model = model
        self.name = name
        self.stabilizations = 0  # # of internal errors ignored
        self.fallback_preds = 0  # # of prediction fallbacks

    def _safe_call(self, fn, *args, **kwargs):
        """
        Execute internal routines with protection. We ignore:
        - IndexError (rule structure misalignment)
        - ValueError / FloatingPointError (numeric issues)
        """
        try:
            return fn(*args, **kwargs)

        except (IndexError, ValueError, FloatingPointError):
            self.stabilizations += 1
            return None

    def fit(self, X, y):
        return self._safe_call(self.model.fit, X, y)

    def evolve(self, X, y):
        if hasattr(self.model, "evolve"):
            return self._safe_call(self.model.evolve, X, y)
        return self.fit(X, y)

    def predict(self, X):
        try:
            y = self.model.predict(X)
            y = np.asarray(y).ravel()
            if not np.isfinite(y).all():
                raise FloatingPointError
            return y
        except Exception:
            # fallback to class 0
            self.fallback_preds += 1
            return np.zeros(len(X), dtype=int)

    def n_rules(self):
        try:
            return self.model.n_rules()
        except Exception:
            return -1


# -------------------------------------------------------------------------
# Prequential evaluation for ENFSUni0 (NF-type API)
# -------------------------------------------------------------------------


def prequential_nf_generic(model, X, y, desc="NF") -> Dict[str, Any]:
    """
    Prequential evaluation for ENFSUni0Evolving.

    Expected API:
      - predict(X)  -> label(s)
      - partial_fit(X, y)
      - attribute: n_rls_resets (count of RLS instabilities repaired)
    """
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)
    rules = []

    pred_fallbacks = 0  # if we need to force class 0 by NaN protection

    for t in tqdm(range(n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

        if t == 0:
            y_hat = 0
        else:
            try:
                yhat_raw = np.asarray(model.predict(x_t)).ravel()
                if yhat_raw.size == 0 or not np.isfinite(yhat_raw[0]):
                    raise FloatingPointError
                y_hat = int(yhat_raw[0])
            except Exception:
                # fallback prediction
                y_hat = 0
                pred_fallbacks += 1

        y_pred[t] = y_hat
        correct[t] = 1.0 if y_hat == yt else 0.0
        acc[t] = correct[: t + 1].mean()

        model.partial_fit(x_t, np.array([yt]))

        # track rules over time
        if hasattr(model, "history_n_rules") and model.history_n_rules:
            rules.append(model.history_n_rules[-1])
        elif hasattr(model, "n_rules_") and model.n_rules_ is not None:
            rules.append(model.n_rules_)
        elif hasattr(model, "rules"):
            rules.append(len(model.rules))
        else:
            rules.append(np.nan)

    rules_arr = np.array(rules, float)
    n_rules_final = int(rules_arr[-1]) if not np.isnan(rules_arr[-1]) else -1

    # for ENFSUni0 we count RLS instabilities internally
    P_instabilities = getattr(model, "n_rls_resets", 0)

    return dict(
        accuracy=acc,
        y_pred=y_pred,
        n_rules_time=rules_arr,
        n_rules_final=n_rules_final,
        pred_fallbacks=int(pred_fallbacks),
        P_instabilities=int(P_instabilities),
    )


# -------------------------------------------------------------------------
# Prequential evaluation for Kaike EFS models (with SafeEFSWrapper)
# -------------------------------------------------------------------------


def prequential_efs(model, X, y, warmup=200, batch_evolve=20, desc="eFS"):
    """
    Prequential evaluation for Kaike eFS models.

    Strategy:
      - Warmup: first `warmup` samples -> fit()
      - Then, for each incoming sample t >= warmup:
          * predict
          * accumulate prequential accuracy
          * buffer sample
          * when buffer reaches `batch_evolve`, call model.fit on the block

    Expected API:
      - fit(X, y)
      - predict(X) -> labels or scores
      - n_rules()  -> number of rules (if available)
      - (if wrapped by SafeEFSWrapper) attributes:
            .fallback_preds
            .stabilizations
    """
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)

    if warmup >= n:
        warmup = max(1, n // 10)

    # warmup training
    model.fit(X[:warmup], y[:warmup])

    bufX = []
    bufY = []

    for t in tqdm(range(warmup, n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

        yhat_raw = model.predict(x_t)  # wrapper already handles NaN
        yhat_raw = np.asarray(yhat_raw).ravel()

        if yhat_raw.size == 0:
            yh = 0
        else:
            # if probabilistic in [0,1], threshold at 0.5
            if 0 <= yhat_raw[0] <= 1:
                yh = int(yhat_raw[0] >= 0.5)
            else:
                yh = int(yhat_raw[0])

        y_pred[t] = yh
        correct[t] = 1.0 if yh == yt else 0.0
        acc[t] = correct[: t + 1].mean()

        bufX.append(x_t.ravel())
        bufY.append(yt)

        if len(bufY) >= batch_evolve:
            X_block = np.vstack(bufX)
            y_block = np.array(bufY, int)
            model.fit(X_block, y_block)
            bufX.clear()
            bufY.clear()

    # define accuracy for warmup segment as the first post-warmup value
    if n > warmup:
        acc[:warmup] = acc[warmup]

    try:
        n_rules_final = int(model.n_rules())
    except Exception:
        n_rules_final = -1

    rules_time = np.full_like(acc, n_rules_final, float)

    pred_fallbacks = int(getattr(model, "fallback_preds", 0))
    P_instabilities = int(getattr(model, "stabilizations", 0))

    return dict(
        accuracy=acc,
        y_pred=y_pred,
        n_rules_time=rules_time,
        n_rules_final=n_rules_final,
        pred_fallbacks=pred_fallbacks,
        P_instabilities=P_instabilities,
    )


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------


def plot_multi_accuracy(
    acc_dict, drift_points, dataset, feature_mode, drift_mode, outdir: Path
):
    """Plot accuracy curves of all models in one figure with drift markers."""
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, acc in acc_dict.items():
        plt.plot(acc, label=name)

    for dp in drift_points:
        plt.axvline(dp, color="red", linestyle="--", alpha=0.6)
        plt.scatter([dp], [0], color="red", marker="x")

    plt.ylim(0, 1)
    plt.xlabel("Samples")
    plt.ylabel("Prequential Accuracy")
    plt.title(f"{dataset} – features={feature_mode}, drifts={drift_mode}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    fname = f"acc_all_{dataset}_{feature_mode}_{drift_mode}.png"
    plt.savefig(outdir / fname, dpi=300)
    plt.close()


def plot_single_model(
    acc, rules, model, dataset, feature_mode, drift_points, outdir: Path
):
    """Per-model plots: accuracy and rule evolution."""
    outdir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(acc))

    # accuracy
    plt.figure(figsize=(7, 4))
    plt.plot(t, acc)
    for dp in drift_points:
        plt.axvline(dp, color="red", linestyle="--", alpha=0.5)
        plt.scatter([dp], [0], color="red", marker="x")

    plt.ylim(0, 1)
    plt.xlabel("Samples")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset} – {model} – {feature_mode}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"acc_{model}_{feature_mode}.png", dpi=300)
    plt.close()

    # rules
    if rules is not None and not np.all(np.isnan(rules)):
        plt.figure(figsize=(7, 4))
        plt.plot(t, rules)
        for dp in drift_points:
            plt.axvline(dp, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("Samples")
        plt.ylabel("#Rules")
        plt.title(f"{dataset} – {model} – rule evolution – {feature_mode}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"rules_{model}_{feature_mode}.png", dpi=300)
        plt.close()


# -------------------------------------------------------------------------
# LaTeX table
# -------------------------------------------------------------------------


def build_latex_table(summary, outpath: Path):
    """
    Build LaTeX summary table.

    Columns:
      Features, Drift, Model, Acc_final, Acc_mean, Rules_final,
      Pred_fallbacks, P_instabilities
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{Prequential accuracy, rule growth and numeric stability "
        "indicators on the Apache due-date stream.}"
    )
    lines.append("\\begin{tabular}{lllcccccc} \\toprule")
    lines.append(
        "Features & Drift & Model & Acc$_{final}$ & Acc$_{mean}$ & "
        "$N_{rules}^{final}$ & Fallbacks & Instabilities \\\\ \\midrule"
    )

    for feature_mode in FEATURE_MODES:
        for drift_mode in DRIFT_MODES:
            for model in MODEL_NAMES:
                key = (feature_mode, drift_mode, model)
                if key not in summary:
                    continue
                s = summary[key]
                lines.append(
                    f"{feature_mode} & {drift_mode} & {model} & "
                    f"{s['acc_final']:.3f} & {s['acc_mean']:.3f} & "
                    f"{int(s['n_rules_final'])} & "
                    f"{int(s['pred_fallbacks'])} & "
                    f"{int(s['P_instabilities'])} \\\\"
                )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_name = "Apache_due"

    summary = {}

    from sklearn.preprocessing import StandardScaler

    for feature_mode in FEATURE_MODES:
        print(f"\n=== Feature mode: {feature_mode} ===")

        X, y, feat_names = load_apache_numeric(feature_mode)
        n = len(y)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        fixed3 = fixed_drifts(n, 3)
        real3 = detect_real_drifts_from_y(y, 3)

        for drift_mode in DRIFT_MODES:
            print(f"--- Drift mode: {drift_mode}")

            drift_points = fixed3 if drift_mode == "fixed3" else real3

            outdir = RESULTS_ROOT / feature_mode / drift_mode
            outdir.mkdir(parents=True, exist_ok=True)

            acc_dict = {}

            for model_name in MODEL_NAMES:
                print(f"Running model: {model_name}")

                # ------------------------------------------------------------------
                # Instantiate model
                # ------------------------------------------------------------------
                if model_name == "ENFSUni0":
                    model = ENFSUni0Evolving(
                        n_features=X.shape[1],
                        n_classes=2,
                        lambda_ff=0.99,
                        q0=1.0,
                        sim_threshold=0.95,
                        max_rules=10,
                        random_state=42,
                    )
                    res = prequential_nf_generic(
                        model,
                        Xs,
                        y,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                elif model_name == "ePL":
                    model = ePL()
                    res = prequential_efs(
                        model,
                        Xs,
                        y,
                        WARMUP,
                        BATCH_EVOLVE,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                elif model_name == "ePL_plus":
                    model = SafeEFSWrapper(ePL_plus(), name="ePL_plus")
                    res = prequential_efs(
                        model,
                        Xs,
                        y,
                        WARMUP,
                        BATCH_EVOLVE,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                elif model_name == "exTS":
                    model = SafeEFSWrapper(exTS(), name="exTS")
                    res = prequential_efs(
                        model,
                        Xs,
                        y,
                        WARMUP,
                        BATCH_EVOLVE,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                elif model_name == "Simpl_eTS":
                    model = SafeEFSWrapper(Simpl_eTS(), name="Simpl_eTS")
                    res = prequential_efs(
                        model,
                        Xs,
                        y,
                        WARMUP,
                        BATCH_EVOLVE,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                elif model_name == "eMG":
                    model = SafeEFSWrapper(eMG(), name="eMG")
                    res = prequential_efs(
                        model,
                        Xs,
                        y,
                        WARMUP,
                        BATCH_EVOLVE,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                elif model_name == "ePL_KRLS_DISCO":
                    model = SafeEFSWrapper(
                        ePL_KRLS_DISCO(
                            alpha=0.001,
                            beta=0.05,
                            sigma=0.5,
                            lambda1=0.0000001,
                            e_utility=0.05,
                            tau=0.05,
                            omega=1,
                        ),
                        name="ePL_KRLS_DISCO",
                    )
                    res = prequential_efs(
                        model,
                        Xs,
                        y,
                        WARMUP,
                        BATCH_EVOLVE,
                        desc=f"{model_name} ({feature_mode}, {drift_mode})",
                    )

                else:
                    raise ValueError(model_name)

                # ------------------------------------------------------------------
                # Collect metrics
                # ------------------------------------------------------------------
                acc = res["accuracy"]
                rules = res["n_rules_time"]
                n_rules_final = res["n_rules_final"]
                pred_fallbacks = res.get("pred_fallbacks", 0)
                P_instabilities = res.get("P_instabilities", 0)

                acc_dict[model_name] = acc

                # Save to summary dict
                summary[(feature_mode, drift_mode, model_name)] = dict(
                    acc_final=float(acc[-1]),
                    acc_mean=float(acc.mean()),
                    n_rules_final=float(n_rules_final),
                    pred_fallbacks=int(pred_fallbacks),
                    P_instabilities=int(P_instabilities),
                )

                # Per-model plots
                plot_single_model(
                    acc,
                    rules,
                    model_name,
                    dataset_name,
                    feature_mode,
                    drift_points,
                    outdir,
                )

                # Console log
                print(
                    f"[RESULT] feat={feature_mode:11s} | drift={drift_mode:6s} | "
                    f"model={model_name:14s} | Acc_final={acc[-1]:.4f} | "
                    f"Acc_mean={acc.mean():.4f} | Rules_final={n_rules_final} | "
                    f"Pred_fallbacks={pred_fallbacks} | P_instabilities={P_instabilities}"
                )

            # Multi-model plot for this combination
            plot_multi_accuracy(
                acc_dict, drift_points, dataset_name, feature_mode, drift_mode, outdir
            )

    tex_path = RESULTS_ROOT / "apache_stream_evolving_summary.tex"
    build_latex_table(summary, tex_path)
    print(f"\nLaTeX table saved to: {tex_path}")


if __name__ == "__main__":
    main()

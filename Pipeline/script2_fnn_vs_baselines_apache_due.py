"""
Script 2 – FNN vs. traditional classifiers on Apache 'due' dataset
------------------------------------------------------------------
This script:

  * Loads the numeric dataset produced by Script 0 and the feature set
    selected by Script 1 (features_fnn).
  * Uses the **binary target** `y_binary` (delay vs. non-delay).
  * Runs repeated 10-fold *experiment runs* (different random seeds):
      - 70% training / 30% testing split (stratified) for each run.
      - Trains an FNN model and several traditional classifiers:
            - RandomForest
            - MLP (feed-forward neural net)
            - Naive Bayes
            - SVM
      - Computes Accuracy, Precision, Recall and F1-score on the test set.
  * Aggregates the results over runs (mean and standard deviation).
  * Saves:
      - `runs_metrics.csv`  – per run, per model.
      - `summary_metrics.csv` – mean and std for each model.
      - `results_table.tex` – LaTeX table with summary metrics.
      - `model_comparison_accuracy.png` – bar plot (mean ± std) of accuracy.
      - `model_comparison_f1.png` – bar plot (mean ± std) of F1-score.
      - `fnn_rules_seed0.txt` – textual fuzzy rules of the FNN (seed 0).
      - `fnn_rules_seed0.tex` – LaTeX table with a subset of fuzzy rules.

For the mathematical description of the metrics and repeated runs,
see `README_Script2.md`.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# To import FNNModel from the sibling 'models' package
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from models.models import FNNModel  # type: ignore


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_DIR = Path("../apachedataset/processed")  # Script 0 outputs
FEATURE_DIR = DATASET_DIR / "feature_analysis"  # Script 1 outputs

NUMERIC_CSV_NAME = "apache_due_numeric.csv"
SELECTED_FEATURES_JSON = "selected_features.json"

RESULTS_SUBDIR = "fnn_vs_baselines"

N_RUNS = 10
TEST_SIZE = 0.30
RANDOM_SEEDS = list(range(N_RUNS))  # [0, 1, ..., 9]

# FNN hyperparameters (can be adjusted later / moved to a JSON config)
FNN_NUM_MFS = 4
FNN_NEURON_TYPE = "orneuron"
FNN_ACTIVATION = "sign"
FNN_OPTIMIZER = "moore-penrose"
FNN_PRUNING = "none"


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


from typing import List  # já tem


def load_data_and_features() -> tuple[np.ndarray, np.ndarray, List[str]]:
    ...

    """Load numeric dataset and selected FNN features.

    Returns
    -------
    X : np.ndarray
        Feature matrix (all rows, selected columns).
    y : np.ndarray
        Binary target array (0 = no delay, 1 = delayed).
    feature_names : list of str
        Names of the selected input features (for interpretability and logging).
    """
    numeric_csv_path = DATASET_DIR / NUMERIC_CSV_NAME
    selected_features_path = FEATURE_DIR / SELECTED_FEATURES_JSON

    logger.info("Loading numeric dataset from %s", numeric_csv_path)
    df = pd.read_csv(numeric_csv_path)

    logger.info("Loading selected features from %s", selected_features_path)
    with open(selected_features_path, "r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    feature_names = feature_sets["features_fnn"]
    logger.info(
        "Using %d features for FNN and baselines: %s", len(feature_names), feature_names
    )

    X = df[feature_names].values.astype(float)
    y = df["y_binary"].values.astype(int)

    return X, y, feature_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1-score (binary classification).

    This is used for *all* models so that metrics are comparable.
    See README_Script2.md, Section 3 for the formal definitions.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_and_evaluate_baselines(
    X_train_raw, X_test_raw, y_train, y_test, seed: int
) -> Dict[str, Dict[str, float]]:
    """Train traditional classifiers and evaluate them on the test set.

    Baselines are trained on *standardised* features, while the FNN uses
    the raw features (to preserve interpretability in the original space).
    """
    results = {}

    # Standardisation for baseline models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            max_iter=500,
            random_state=seed,
        ),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=seed,
        ),
    }

    for model_name, model in models.items():
        logger.info("  Training baseline model: %s (seed=%d)", model_name, seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = compute_metrics(y_test, y_pred)

    return results


def train_and_evaluate_fnn(X_train_raw, X_test_raw, y_train, y_test, seed: int):
    """Train the FNN model and evaluate on the test set.

    Note: FNNModel expects labels in {-1, +1}. We convert from {0, 1}.
    """
    logger.info("  Training FNN model (seed=%d)...", seed)

    # Convert labels 0/1 to -1/+1 for the FNN
    y_train_signed = np.where(y_train == 1, 1, -1)
    y_test_signed = np.where(y_test == 1, 1, -1)

    # rng_seed can be any hashable seed; we simply pass the integer seed
    rng = np.random.default_rng(seed)

    fnn_model = FNNModel(
        num_mfs=FNN_NUM_MFS,
        neuron_type=FNN_NEURON_TYPE,
        activation=FNN_ACTIVATION,
        optimizer=FNN_OPTIMIZER,
        pruning=FNN_PRUNING,
        visualizeMF=False,
        rng_seed=rng,  # agora é um Generator, com .random()
    )

    fnn_model.train_model(X_train_raw, y_train_signed)
    y_pred_signed, _ = fnn_model.evaluate_model(X_test_raw, y_test_signed)

    # Convert back to {0,1}
    y_pred = np.where(y_pred_signed > 0, 1, 0)
    metrics = compute_metrics(y_test, y_pred)

    return fnn_model, metrics, y_pred


def save_fnn_rules(fnn_model, output_dir: Path, max_rules_for_tex: int = 15) -> None:
    """Generate fuzzy rules from the trained FNN and save them.

    Parameters
    ----------
    fnn_model : FNNModel
        Trained FNN model.
    output_dir : Path
        Directory where the rule files will be written.
    max_rules_for_tex : int
        Maximum number of rules to include in the LaTeX table (for readability).
    """
    logger.info("Generating and saving fuzzy rules for interpretation...")
    rules = fnn_model.generate_fuzzy_rules()

    # Plain-text version (all rules)
    txt_path = output_dir / "fnn_rules_seed0.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(r + "\n")

    # LaTeX table with a subset of rules
    tex_path = output_dir / "fnn_rules_seed0.tex"
    rules_subset = rules[:max_rules_for_tex]

    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% FNN fuzzy rules (subset) automatically generated by Script 2\n")
        f.write("% For a complete list, see fnn_rules_seed0.txt\n\n")
        f.write("\\begin{table}[ht]\\centering\n")
        f.write(
            "\\caption{Example of fuzzy rules induced by the FNN model (Apache due-date dataset, seed 0).}\\label{tab:fnn-rules-apache}\n"
        )
        f.write("\\begin{tabular}{p{0.95\\linewidth}}\\toprule\n")
        for r in rules_subset:
            safe_rule = r.replace("_", "\\_")  # basic escaping
            f.write(safe_rule + "\\\\\\midrule\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    logger.info("Saved fuzzy rules to %s and %s", txt_path, tex_path)


def aggregate_results(runs_records: List[Dict]) -> pd.DataFrame:
    """Aggregate metrics across runs for each model.

    Parameters
    ----------
    runs_records : list of dict
        Each element is of the form::

            {"seed": int, "model": str, "accuracy": float, ...}
    """
    df = pd.DataFrame(runs_records)
    grouped = df.groupby("model")

    summary_rows = []
    for model_name, group in grouped:
        row = {"model": model_name}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean_val = group[metric].mean()
            std_val = group[metric].std(ddof=1)
            row[f"{metric}_mean"] = float(mean_val)
            row[f"{metric}_std"] = float(std_val)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("model").sort_index()
    return summary_df


def summary_to_latex(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Export summary metrics to a LaTeX table."""
    df = summary_df.copy()
    # Round values for readability
    for col in df.columns:
        df[col] = df[col].round(3)

    latex = df.to_latex(
        caption=(
            "Mean and standard deviation (over 10 runs) of classification "
            "metrics for FNN and baseline models on the Apache due-date dataset "
            "(binary target: delayed vs. not delayed)."
        ),
        label="tab:results-fnn-vs-baselines-apache",
        escape=False,
    )

    output_path.write_text(latex, encoding="utf-8")
    logger.info("Saved LaTeX summary table to %s", output_path)


def plot_bar_with_error(
    summary_df: pd.DataFrame, metric: str, output_path: Path
) -> None:
    """Create a bar plot with error bars for a given metric.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame indexed by model with columns like `<metric>_mean` and
        `<metric>_std`.
    metric : str
        Name of the metric (`"accuracy"`, `"f1"`, etc.).
    output_path : Path
        Where to save the PNG file.
    """
    means = summary_df[f"{metric}_mean"]
    stds = summary_df[f"{metric}_std"]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary_df.index))
    plt.bar(x, means.values, yerr=stds.values, capsize=5)
    plt.xticks(x, summary_df.index, rotation=30)
    plt.ylabel(metric.capitalize())
    plt.ylim(0.0, 1.0)
    plt.title(f"Model comparison – {metric.capitalize()}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved bar plot for %s to %s", metric, output_path)


# -------------------------------------------------------------------------
# Main experiment loop
# -------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting Script 2: FNN vs. baseline classifiers.")

    # Resolve paths
    results_dir = FEATURE_DIR / RESULTS_SUBDIR
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load data and selected features
    X, y, feature_names = load_data_and_features()
    n_samples, n_features = X.shape
    logger.info("Dataset ready: %d samples, %d features.", n_samples, n_features)

    runs_records = []  # will collect metrics for each (model, seed)
    fnn_model_seed0 = None

    # Loop over random seeds (runs)
    for seed in tqdm(RANDOM_SEEDS, desc="Runs"):
        logger.info("=== Run with seed=%d ===", seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        # FNN (uses raw features)
        fnn_model, fnn_metrics, _ = train_and_evaluate_fnn(
            X_train, X_test, y_train, y_test, seed
        )
        if seed == 0:
            fnn_model_seed0 = fnn_model
        record_fnn = {"seed": seed, "model": "FNN"}
        record_fnn.update(fnn_metrics)
        runs_records.append(record_fnn)

        # Baselines (use standardised features)
        baseline_results = train_and_evaluate_baselines(
            X_train, X_test, y_train, y_test, seed
        )
        for model_name, metrics in baseline_results.items():
            record = {"seed": seed, "model": model_name}
            record.update(metrics)
            runs_records.append(record)

    # Convert to DataFrame and save per-run metrics
    runs_df = pd.DataFrame(runs_records)
    runs_csv_path = results_dir / "runs_metrics.csv"
    runs_df.to_csv(runs_csv_path, index=False)
    logger.info("Saved per-run metrics to %s", runs_csv_path)

    # Aggregate results
    summary_df = aggregate_results(runs_records)
    summary_csv_path = results_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path)
    logger.info("Saved summary metrics to %s", summary_csv_path)

    # Save LaTeX results table
    latex_path = results_dir / "results_table.tex"
    summary_to_latex(summary_df, latex_path)

    # Plots: accuracy and F1-score comparisons
    plot_bar_with_error(
        summary_df, "accuracy", results_dir / "model_comparison_accuracy.png"
    )
    plot_bar_with_error(summary_df, "f1", results_dir / "model_comparison_f1.png")

    # Save FNN rules for interpretability (using the model from seed 0)
    if fnn_model_seed0 is not None:
        save_fnn_rules(fnn_model_seed0, results_dir)
    else:
        logger.warning("No FNN model from seed 0 available; skipping rule export.")

    logger.info("Script 2 finished successfully.")


if __name__ == "__main__":
    main()

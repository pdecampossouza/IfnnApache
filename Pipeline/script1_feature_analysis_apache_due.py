"""
Script 1 – Feature analysis for Apache 'due' dataset
----------------------------------------------------
This script consumes the *numeric* dataset produced by Script 0 and:

  1. Identifies candidate feature columns for modelling.
  2. Computes basic statistics for each feature (mean, std, min, max).
  3. Computes the Pearson linear correlation between each feature and
     the continuous delay variable `delaydays`.
  4. Trains a RandomForest classifier (multi-class, target = impact_class)
     to estimate feature importances (Gini importance).
  5. Creates a derived feature `num_deps` = no_issuelink + no_blocking
     + no_blockedby (interpretable as "number of dependencies").
  6. Proposes two ordered feature lists:
     - `features_fnn`  : up to 8 interpretable features for the FNN model.
     - `features_enfs` : a richer set (up to 20 features) for the eFNN model.
  7. Saves:
     - `feature_stats.csv` with per-feature statistics.
     - `selected_features.json` with the proposed feature sets.
     - Figures for correlation and importance analysis.
     - LaTeX tables for direct inclusion in the thesis/paper.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_DIR = Path("../apachedataset/processed")  # where Script 0 wrote outputs
NUMERIC_CSV_NAME = "apache_due_numeric.csv"
METADATA_JSON_NAME = "apache_due_metadata.json"

OUTPUT_SUBDIR = "feature_analysis"

FEATURE_STATS_CSV = "feature_stats.csv"
SELECTED_FEATURES_JSON = "selected_features.json"

FIG_IMPORTANCE_NAME = "feature_importance_rf.png"
FIG_CORR_HEATMAP_NAME = "feature_correlation_heatmap.png"

TABLE_TOP10_NAME = "table_top10_features.tex"
TABLE_FNN_NAME = "table_fnn_features.tex"


# -------------------------------------------------------------------------
# Logging setup
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


def get_candidate_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numeric candidate feature columns.

    We exclude:
      - identifiers       (issuekey)
      - raw categorical   (type, priority)
      - datetime          (openeddate)
      - explicit targets  (impact_class, y_binary)

    All remaining numeric columns are considered as potential features.
    """
    exclude_cols = {
        "issuekey",
        "type",
        "priority",
        "openeddate",
        "impact_class",
        "y_binary",
    }
    candidates = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    return candidates


def compute_feature_statistics(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Compute basic statistics and correlation with delaydays for each feature.

    Statistics:
      - mean, std, min, max
      - Pearson correlation with delaydays
    """
    stats = []
    delay = df["delaydays"].astype(float)

    logger.info(
        "Computing per-feature statistics and correlations (with progress bar)..."
    )
    for col in tqdm(feature_cols, desc="Features"):
        series = df[col].astype(float)
        if series.std() > 0:
            corr = np.corrcoef(series, delay)[0, 1]
        else:
            corr = 0.0
        stats.append(
            {
                "feature": col,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "corr_with_delaydays": float(corr),
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("feature").reset_index(drop=True)
    return stats_df


def compute_rf_importance(df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """Estimate feature importances using a RandomForest classifier.

    Target: impact_class (4-class variable).

    The RF is not meant to be the final model; it serves only as a
    *supervised ranking* of features.
    """
    logger.info(
        "Training RandomForest for feature importance (this may take a few seconds)..."
    )

    X = df[feature_cols].values
    y = df["impact_class"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_scaled, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    return importances


def create_interpretable_feature_sets(importances: pd.Series) -> Dict[str, List[str]]:
    """Create ordered feature lists for FNN and eFNN models.

    We start from a *manually curated* list of interpretable variables
    (column names). Then we intersect it with the ranking given by the RF
    importance, so that the final order still reflects empirical relevance.
    """
    interpretable_pool = [
        "RemainingDay",
        "ProgressTime",
        "perofdelay",
        "workload",
        "priority_level",
        "no_comment",
        "no_priority_change",
        "num_deps",
        "discussion",
        "repetition",
        "reporterrep",
        "opened_year",
        "opened_month",
    ]

    ordered_interpretable = [f for f in importances.index if f in interpretable_pool]

    # FNN: at most 6 features
    features_fnn = ordered_interpretable[:6]

    # eFNN: richer set, starting from interpretable and adding more
    features_enfs = list(ordered_interpretable)
    for f in importances.index:
        if f not in features_enfs:
            features_enfs.append(f)
        if len(features_enfs) >= 20:
            break

    return {
        "features_fnn": features_fnn,
        "features_enfs": features_enfs,
    }


def plot_importances(importances: pd.Series, output_dir: Path) -> None:
    """Plot a bar chart of the top 20 features by RF importance."""
    logger.info("Generating feature importance plot...")
    top_k = min(20, len(importances))
    top_importances = importances.iloc[:top_k]

    plt.figure(figsize=(10, 6))
    top_importances[::-1].plot(kind="barh")  # reverse to show most important at top
    plt.xlabel("RandomForest Gini importance")
    plt.title("Top features according to RandomForest (impact_class)")
    plt.tight_layout()
    fig_path = output_dir / FIG_IMPORTANCE_NAME
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved RF importance figure to %s", fig_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, feature_cols: List[str], output_dir: Path
) -> None:
    """Plot a correlation heatmap for a subset of features.

    To keep the figure readable, we select up to the first 15 features
    from the given list (already in some meaningful order, usually by
    importance).
    """
    logger.info("Generating correlation heatmap...")
    max_features = min(15, len(feature_cols))
    cols_subset = feature_cols[:max_features]

    corr = df[cols_subset].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (subset of features)")
    plt.tight_layout()
    fig_path = output_dir / FIG_CORR_HEATMAP_NAME
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved correlation heatmap to %s", fig_path)


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert a DataFrame with numeric statistics to a LaTeX table string.

    Columns are rounded for readability.
    """
    cols = [
        "feature",
        "mean",
        "std",
        "min",
        "max",
        "corr_with_delaydays",
        "rf_importance",
    ]
    df2 = df[cols].copy()
    df2["mean"] = df2["mean"].round(3)
    df2["std"] = df2["std"].round(3)
    df2["min"] = df2["min"].round(3)
    df2["max"] = df2["max"].round(3)
    df2["corr_with_delaydays"] = df2["corr_with_delaydays"].round(3)
    df2["rf_importance"] = df2["rf_importance"].round(4)
    latex = df2.to_latex(index=False, caption=caption, label=label, escape=False)
    return latex


def save_latex_tables(
    feature_stats: pd.DataFrame, feature_sets: Dict[str, List[str]], output_dir: Path
) -> None:
    """Generate LaTeX tables for the top-10 features and FNN feature set."""
    logger.info("Generating LaTeX tables for feature statistics...")

    # Top 10 by RF importance
    top10 = feature_stats.sort_values("rf_importance", ascending=False).head(10)
    latex_top10 = df_to_latex_table(
        top10,
        caption="Top 10 features ranked by RandomForest Gini importance (Apache due-date dataset).",
        label="tab:rf-top10-apache",
    )
    (output_dir / TABLE_TOP10_NAME).write_text(latex_top10, encoding="utf-8")

    # FNN features table
    fnn_feats = feature_sets.get("features_fnn", [])
    if fnn_feats:
        fnn_stats = feature_stats.set_index("feature").loc[fnn_feats].reset_index()
        latex_fnn = df_to_latex_table(
            fnn_stats,
            caption="Statistics for the features selected for the FNN model (Apache due-date dataset).",
            label="tab:fnn-features-apache",
        )
        (output_dir / TABLE_FNN_NAME).write_text(latex_fnn, encoding="utf-8")
    logger.info(
        "Saved LaTeX tables to %s and %s",
        output_dir / TABLE_TOP10_NAME,
        output_dir / TABLE_FNN_NAME,
    )


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting Script 1: feature analysis for Apache 'due' dataset.")

    numeric_csv_path = DATASET_DIR / NUMERIC_CSV_NAME
    metadata_json_path = DATASET_DIR / METADATA_JSON_NAME

    output_dir = DATASET_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Step 1/4: Loading numeric dataset and metadata.")
    df = pd.read_csv(numeric_csv_path)
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Numeric dataset: %d rows, %d columns.", df.shape[0], df.shape[1])

    logger.info("Creating derived feature 'num_deps'.")
    df["num_deps"] = (
        df.get("no_issuelink", 0) + df.get("no_blocking", 0) + df.get("no_blockedby", 0)
    )

    logger.info("Step 2/4: Identifying candidate features and computing statistics.")
    candidate_features = get_candidate_feature_columns(df)
    if "num_deps" not in candidate_features:
        candidate_features.append("num_deps")

    feature_stats = compute_feature_statistics(df, candidate_features)

    logger.info("Step 3/4: Computing RandomForest feature importances.")
    importances = compute_rf_importance(df, candidate_features)

    feature_stats = feature_stats.set_index("feature")
    feature_stats["rf_importance"] = importances
    feature_stats = feature_stats.reset_index().sort_values(
        "rf_importance", ascending=False
    )

    stats_csv_path = output_dir / FEATURE_STATS_CSV
    feature_stats.to_csv(stats_csv_path, index=False)
    logger.info("Saved feature statistics to %s", stats_csv_path)

    logger.info(
        "Step 4/4: Building feature sets (FNN / eFNN), saving plots and LaTeX tables."
    )
    feature_sets = create_interpretable_feature_sets(importances)

    selected_features_path = output_dir / SELECTED_FEATURES_JSON
    with open(selected_features_path, "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, indent=4)
    logger.info("Saved selected feature sets to %s", selected_features_path)

    plot_importances(importances, output_dir)
    plot_correlation_heatmap(df, feature_sets["features_fnn"], output_dir)
    save_latex_tables(feature_stats, feature_sets, output_dir)

    logger.info("Script 1 finished successfully.")


if __name__ == "__main__":
    main()

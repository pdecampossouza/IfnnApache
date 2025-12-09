
"""
Script 0 – Data preprocessing for Apache 'due' dataset
------------------------------------------------------
This script:
  1. Loads the raw `apache_due.csv` file.
  2. Converts non‑numeric columns into numeric representations.
  3. Constructs target variables:
     - `impact_class` (4 classes, as in Choetkiertikul et al. 2017)
     - `y_binary` (delay vs. non‑delay)
  4. Saves a numeric CSV and a JSON metadata file.
  5. Generates basic figures (histograms) to inspect delay distribution.

For a mathematical and methodological description of the transformations,
see `README_Script0.md`, Sections 2 and 3.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_DIR = Path("../apachedataset")   # folder that contains apache_due.csv
RAW_CSV_NAME = "apache_due.csv"         # raw input file name
OUTPUT_SUBDIR = "processed"             # where numeric data & figures go

NUMERIC_CSV_NAME = "apache_due_numeric.csv"
METADATA_JSON_NAME = "apache_due_metadata.json"

FIG_DELAY_HIST_NAME = "delaydays_histogram.png"
FIG_IMPACT_HIST_NAME = "impact_class_histogram.png"


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

def encode_categorical_column(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    """Map a categorical string column to integer codes using a given mapping."""
    return series.map(mapping).fillna(0).astype(int)


def compute_impact_classes(delay: pd.Series, p_low: float = 33.3, p_high: float = 66.6):
    """Compute 4-class impact variable from `delaydays`.

    See README_Script0.md, Section 3.1.
    """
    delay = delay.astype(float)
    positive_delays = delay[delay > 0]
    if positive_delays.empty:
        raise ValueError("No positive delays found; cannot compute impact classes.")
    T1 = np.percentile(positive_delays, p_low)
    T2 = np.percentile(positive_delays, p_high)
    impact_class = pd.Series(0, index=delay.index, dtype=int)
    impact_class[(delay > 0) & (delay <= T1)] = 1
    impact_class[(delay > T1) & (delay <= T2)] = 2
    impact_class[delay > T2] = 3
    thresholds = {"T1": float(T1), "T2": float(T2), "p_low": p_low, "p_high": p_high}
    return impact_class, thresholds


def generate_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save histograms for `delaydays` and `impact_class`."""
    logger.info("Generating histograms...")

    plt.figure()
    df["delaydays"].plot(kind="hist", bins=50)
    plt.xlabel("Delay in days")
    plt.ylabel("Frequency")
    plt.title("Distribution of delaydays")
    fig_path_delay = output_dir / FIG_DELAY_HIST_NAME
    plt.tight_layout()
    plt.savefig(fig_path_delay)
    plt.close()

    plt.figure()
    df["impact_class"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Impact class (0: no delay, 1: small, 2: medium, 3: large)")
    plt.ylabel("Count")
    plt.title("Distribution of impact classes")
    fig_path_impact = output_dir / FIG_IMPACT_HIST_NAME
    plt.tight_layout()
    plt.savefig(fig_path_impact)
    plt.close()

    logger.info("Saved histogram of delaydays to %s", fig_path_delay)
    logger.info("Saved histogram of impact_class to %s", fig_path_impact)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting Script 0: preprocessing raw Apache 'due' dataset.")

    raw_csv_path = DATASET_DIR / RAW_CSV_NAME
    output_dir = DATASET_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(exist_ok=True, parents=True)

    numeric_csv_path = output_dir / NUMERIC_CSV_NAME
    metadata_json_path = output_dir / METADATA_JSON_NAME

    # Step 1: Load data
    logger.info("Step 1/5: Loading raw data from %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)
    logger.info("Loaded %d rows and %d columns.", df.shape[0], df.shape[1])

    # Step 2: Convert openeddate and delaydays
    logger.info("Step 2/5: Converting openeddate to datetime and ensuring numeric delaydays.")
    df["openeddate"] = pd.to_datetime(df["openeddate"], errors="coerce")
    df["opened_year"] = df["openeddate"].dt.year
    df["opened_month"] = df["openeddate"].dt.month
    df["delaydays"] = pd.to_numeric(df["delaydays"], errors="coerce").fillna(0).astype(float)

    # Step 3: Encode categorical variables
    logger.info("Step 3/5: Encoding categorical variables (type, priority).") 
    type_categories = sorted(df["type"].dropna().unique().tolist())
    type_mapping = {cat: i + 1 for i, cat in enumerate(type_categories)}
    priority_categories = sorted(df["priority"].dropna().unique().tolist())
    priority_mapping = {cat: i + 1 for i, cat in enumerate(priority_categories)}
    df["type_code"] = encode_categorical_column(df["type"], type_mapping)
    df["priority_level"] = encode_categorical_column(df["priority"], priority_mapping)

    # Step 4: Targets
    logger.info("Step 4/5: Computing impact classes and binary target.")
    impact_class, thresholds = compute_impact_classes(df["delaydays"])
    df["impact_class"] = impact_class
    df["y_binary"] = (df["delaydays"] > 0).astype(int)

    # Step 5: Save outputs
    logger.info("Step 5/5: Saving numeric dataset and metadata.")
    df.to_csv(numeric_csv_path, index=False)

    import json
    metadata = {
        "raw_csv_path": str(raw_csv_path),
        "numeric_csv_path": str(numeric_csv_path),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "type_mapping": type_mapping,
        "priority_mapping": priority_mapping,
        "delaydays_thresholds": thresholds,
        "targets": {
            "impact_class": "4-class impact variable derived from delaydays (see README_Script0.md, Section 3.1)",
            "y_binary": "1 if delaydays > 0, else 0",
        },
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    generate_histograms(df, output_dir)
    logger.info("Script 0 finished successfully.")


if __name__ == "__main__":
    main()

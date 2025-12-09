"""
Script 2 – FNN vs. traditional classifiers on Apache 'due' dataset (flexible optimizer)
------------------------------------------------------------------
Versão flexível para diferentes otimizadores do FNN.

Regras:
  * Se FNN_OPTIMIZER in {"moore-penrose"}  -> rótulos em {-1, +1}, activation="linear".
  * Se FNN_OPTIMIZER in {"adam", "pso"}   -> rótulos em {0, 1},   activation="sigmoid".

O restante do protocolo é igual à versão normalizada:
  * lê apache_due_numeric.csv e selected_features.json
  * 10 seeds, split estratificado 70/30
  * MESMA normalização (StandardScaler) para FNN e baselines
  * calcula accuracy, precision, recall, F1
  * salva CSVs, LaTeX e figuras
  * extrai regras fuzzy do FNN (usando o modelo da seed 0)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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

# Opcional: reduzir logs C++ do TensorFlow, se estiver instalado
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Importar FNNModel a partir da pasta 'models' (um nível acima de Pipeline)
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from models.models import FNNModel  # type: ignore


# -------------------------------------------------------------------------
# Configuração
# -------------------------------------------------------------------------

DATASET_DIR = Path("../apachedataset/processed")  # saída Script 0
FEATURE_DIR = DATASET_DIR / "feature_analysis"  # saída Script 1

NUMERIC_CSV_NAME = "apache_due_numeric.csv"
SELECTED_FEATURES_JSON = "selected_features.json"

RESULTS_SUBDIR = "fnn_vs_baselines_flexopt"

N_RUNS = 10
TEST_SIZE = 0.30
RANDOM_SEEDS = list(range(N_RUNS))  # [0, 1, ..., 9]

# >>> AQUI você escolhe o otimizador do FNN para este experimento <<<
# Opções esperadas pelo seu FNNModel: "moore-penrose", "adam", "pso"
FNN_OPTIMIZER = "moore-penrose"

FNN_NUM_MFS = 4
FNN_NEURON_TYPE = "orneuron"
# FNN_ACTIVATION será decidido automaticamente com base no otimizador.


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
# Funções auxiliares
# -------------------------------------------------------------------------


def load_data_and_features() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carrega dataset numérico e features selecionadas para o FNN."""
    numeric_csv_path = DATASET_DIR / NUMERIC_CSV_NAME
    selected_features_path = FEATURE_DIR / SELECTED_FEATURES_JSON

    logger.info("Loading numeric dataset from %s", numeric_csv_path)
    df = pd.read_csv(numeric_csv_path)

    logger.info("Loading selected features from %s", selected_features_path)
    with open(selected_features_path, "r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    feature_names = feature_sets["features_fnn"]
    logger.info(
        "Using %d features for all models: %s", len(feature_names), feature_names
    )

    X = df[feature_names].values.astype(float)
    y = df["y_binary"].values.astype(int)  # 0 = no delay, 1 = delay

    return X, y, feature_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy, Precision, Recall, F1 (binário)."""
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
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Treina RF, MLP, NB, SVM nos dados NORMALIZADOS."""
    results: Dict[str, Dict[str, float]] = {}

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


def decide_fnn_label_mode_and_activation(optimizer: str) -> Tuple[str, str]:
    """Define modo de rótulos e activation do FNN a partir do otimizador.

    Retorna
    -------
    label_mode : {"signed", "binary"}
        - "signed"  -> usa rótulos em {-1, +1}
        - "binary"  -> usa rótulos em {0, 1}
    activation : str
        String a ser passada para o parâmetro `activation` do FNNModel.
    """
    if optimizer == "moore-penrose":
        # Regressão linear com sinal: ideal em {-1,+1} e saída linear.
        return "signed", "linear"
    elif optimizer in ("adam", "pso"):
        # Otimizadores de rede: tratam como classificação 0/1 (sigmoid na saída conceitual).
        return "binary", "sigmoid"
    else:
        logger.warning(
            "Unknown optimizer '%s'. Using default label_mode='binary', activation='linear'.",
            optimizer,
        )
        return "binary", "linear"


def train_and_evaluate_fnn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
):
    """Treina o FNN com o otimizador escolhido e avalia no conjunto de teste."""
    label_mode, activation = decide_fnn_label_mode_and_activation(FNN_OPTIMIZER)
    logger.info(
        "  Training FNN (optimizer=%s, label_mode=%s, activation=%s, seed=%d)",
        FNN_OPTIMIZER,
        label_mode,
        activation,
        seed,
    )

    # Ajuste dos rótulos conforme o modo
    if label_mode == "signed":
        y_train_for_fnn = np.where(y_train == 1, 1, -1)
        y_test_for_fnn = np.where(y_test == 1, 1, -1)
    else:
        y_train_for_fnn = y_train
        y_test_for_fnn = y_test

    rng = np.random.default_rng(seed)

    fnn_model = FNNModel(
        num_mfs=FNN_NUM_MFS,
        neuron_type=FNN_NEURON_TYPE,
        activation=activation,
        optimizer=FNN_OPTIMIZER,
        pruning="none",
        visualizeMF=False,
        rng_seed=rng,
    )

    fnn_model.train_model(X_train, y_train_for_fnn)
    y_pred_raw, _ = fnn_model.evaluate_model(X_test, y_test_for_fnn)

    # Converter saída do FNN para rótulos 0/1 para cálculo de métricas
    y_pred_raw = np.asarray(y_pred_raw).reshape(-1)

    if label_mode == "signed":
        # saída esperada em {-1,+1} -> 0/1
        y_pred = np.where(y_pred_raw > 0, 1, 0)
    else:
        # saída esperada como probabilidade ou escalar >=0 -> limiar 0.5
        y_pred = np.where(y_pred_raw >= 0.5, 1, 0)

    metrics = compute_metrics(y_test, y_pred)
    return fnn_model, metrics, y_pred


def save_fnn_rules(fnn_model, output_dir: Path, max_rules_for_tex: int = 15) -> None:
    """Gera e salva regras fuzzy do FNN (texto e LaTeX)."""
    logger.info("Generating and saving fuzzy rules for interpretation...")
    rules = fnn_model.generate_fuzzy_rules()

    txt_path = output_dir / "fnn_rules_seed0.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(r + "\n")

    tex_path = output_dir / "fnn_rules_seed0.tex"
    rules_subset = rules[:max_rules_for_tex]

    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% FNN fuzzy rules (subset) automatically generated by Script 2 FLEX\n")
        f.write("% For a complete list, see fnn_rules_seed0.txt\n\n")
        f.write("\begin{table}[ht]\centering\n")
        f.write(
            "\caption{Example of fuzzy rules induced by the FNN model "
            "(Apache due-date dataset, seed 0).}"
            "\label{tab:fnn-rules-apache-flexopt}\n"
        )
        f.write("\begin{tabular}{p{0.95\linewidth}}\toprule\n")
        for r in rules_subset:
            safe_rule = r.replace("_", "\_")
            f.write(safe_rule + "\\\midrule\n")
        f.write("\bottomrule\end{tabular}\end{table}\n")

    logger.info("Saved fuzzy rules to %s and %s", txt_path, tex_path)


def aggregate_results(runs_records: List[Dict]) -> pd.DataFrame:
    """Agrega resultados por modelo (média e desvio)."""
    df = pd.DataFrame(runs_records)
    grouped = df.groupby("model")

    summary_rows = []
    for model_name, group in grouped:
        row: Dict[str, float | str] = {"model": model_name}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean_val = group[metric].mean()
            std_val = group[metric].std(ddof=1)
            row[f"{metric}_mean"] = float(mean_val)
            row[f"{metric}_std"] = float(std_val)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("model").sort_index()
    return summary_df


def summary_to_latex(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Exporta tabela resumo em LaTeX."""
    df = summary_df.copy()
    for col in df.columns:
        df[col] = df[col].round(3)

    latex = df.to_latex(
        caption=(
            "Mean and standard deviation (over 10 runs) of classification "
            "metrics for FNN (optimizer: %s) and baseline models on the "
            "Apache due-date dataset (binary target: delayed vs. not delayed)."
        )
        % FNN_OPTIMIZER,
        label="tab:results-fnn-vs-baselines-apache-flexopt",
        escape=False,
    )

    output_path.write_text(latex, encoding="utf-8")
    logger.info("Saved LaTeX summary table to %s", output_path)


def plot_bar_with_error(
    summary_df: pd.DataFrame, metric: str, output_path: Path
) -> None:
    """Gráfico de barras com barra de erro para uma métrica."""
    means = summary_df[f"{metric}_mean"]
    stds = summary_df[f"{metric}_std"]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary_df.index))
    plt.bar(x, means.values, yerr=stds.values, capsize=5)
    plt.xticks(x, summary_df.index, rotation=30)
    plt.ylabel(metric.capitalize())
    plt.ylim(0.0, 1.0)
    plt.title(f"Model comparison – {metric.capitalize()} (FNN opt: {FNN_OPTIMIZER})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved bar plot for %s to %s", metric, output_path)


# -------------------------------------------------------------------------
# Loop principal de experimento
# -------------------------------------------------------------------------


def main() -> None:
    logger.info(
        "Starting Script 2 FLEX: FNN optimizer=%s (normalised features for all models).",
        FNN_OPTIMIZER,
    )

    results_dir = FEATURE_DIR / RESULTS_SUBDIR / FNN_OPTIMIZER
    results_dir.mkdir(exist_ok=True, parents=True)

    X, y, feature_names = load_data_and_features()
    n_samples, n_features = X.shape
    logger.info("Dataset ready: %d samples, %d features.", n_samples, n_features)

    runs_records: List[Dict] = []
    fnn_model_seed0 = None

    for seed in tqdm(RANDOM_SEEDS, desc="Runs"):
        logger.info("=== Run with seed=%d ===", seed)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        # Normalização única (mesmo scaler para FNN e baselines)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # FNN
        fnn_model, fnn_metrics, _ = train_and_evaluate_fnn(
            X_train, X_test, y_train, y_test, seed
        )
        if seed == 0:
            fnn_model_seed0 = fnn_model
        record_fnn = {"seed": seed, "model": f"FNN_{FNN_OPTIMIZER}"}
        record_fnn.update(fnn_metrics)
        runs_records.append(record_fnn)

        # Baselines
        baseline_results = train_and_evaluate_baselines(
            X_train, X_test, y_train, y_test, seed
        )
        for model_name, metrics in baseline_results.items():
            record = {"seed": seed, "model": model_name}
            record.update(metrics)
            runs_records.append(record)

    runs_df = pd.DataFrame(runs_records)
    runs_csv_path = results_dir / "runs_metrics.csv"
    runs_df.to_csv(runs_csv_path, index=False)
    logger.info("Saved per-run metrics to %s", runs_csv_path)

    summary_df = aggregate_results(runs_records)
    summary_csv_path = results_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path)
    logger.info("Saved summary metrics to %s", summary_csv_path)

    latex_path = results_dir / "results_table.tex"
    summary_to_latex(summary_df, latex_path)

    plot_bar_with_error(
        summary_df, "accuracy", results_dir / "model_comparison_accuracy.png"
    )
    plot_bar_with_error(summary_df, "f1", results_dir / "model_comparison_f1.png")

    if fnn_model_seed0 is not None:
        save_fnn_rules(fnn_model_seed0, results_dir)
    else:
        logger.warning("No FNN model from seed 0 available; skipping rule export.")

    logger.info("Script 2 FLEX finished successfully.")


if __name__ == "__main__":
    main()

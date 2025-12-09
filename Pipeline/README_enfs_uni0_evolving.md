# Script 3 – Stream experiments with evolving fuzzy baselines on Apache "due"

## Objective

This script performs a **streaming / prequential** comparison between the
proposed **ENFSUni0Evolving** model and several **state-of-the-art
evolving fuzzy systems** from the `evolvingfuzzysystems` library on the
Apache "due" dataset.

The main goal is to evaluate how well each method handles **concept
drift** and **non-stationary behaviour** over time, both with a reduced
feature set and with the full numeric feature space.

## Models compared

- `ENFSUni0Evolving` (your evolving neuro-fuzzy classifier)
- `ePL`
- `ePL_plus`
- `exTS`
- `Simpl_eTS`
- `eMG`
- `ePL_KRLS_DISCO` (kernel-based extension with DISCOnnected KRLS)

All models are evaluated under **the same data stream**, with the same
feature representations, drift definitions and normalisation.

## Data and feature modes

The script assumes that **Script 0** and **Script 1** have already been
executed, so that the following files exist:

- `../apachedataset/processed/apache_due_numeric.csv`
- `../apachedataset/processed/feature_analysis/selected_features.json`

Two feature modes are considered:

1. `fnn_selected`  
   Uses the 6 most relevant features selected in Script 1 (FNN-based
   feature importance). This mode is aligned with your **human
   interpretability constraint (≤ 8 dimensions)**.

2. `all_features`  
   Uses all numeric predictors from Script 0, excluding:
   - identifiers: `issuekey`, `openeddate`
   - targets: `delaydays`, `impact_class`, `y_binary`

In both modes, the features are **standardised** using
`sklearn.preprocessing.StandardScaler`.

## Drift modes

Two concept-drift configurations are implemented:

1. `fixed3` – **synthetic drifts**  
   Three drift points at 25%, 50% and 75% of the stream length:

   \[
   d_k = \left\lfloor \frac{k}{4} N \right\rfloor, \quad k = 1,2,3
   \]

2. `real3` – **data-driven drifts**  
   Three drift points estimated from changes in delay rate:

   - Partition the stream into windows of length \( W \).
   - Compute mean delay rate \( \mu_i = \text{mean}(y_{\text{binary}}) \)
     in each window.
   - Compute absolute differences between consecutive windows:
     \(\Delta_i = |\mu_{i+1} - \mu_i|\).
   - Select the three largest \(\Delta_i\); drifts are placed at the
     corresponding window boundaries.

This provides a **realistic notion of drift** based on changes in the
frequency of delayed issues.

## Evaluation protocol

The script implements **prequential evaluation**:

- For each incoming sample \( x_t \):
  1. Predict \( \hat{y}_t \) using the current model.
  2. Update a running prequential accuracy:
     \[
     \text{Acc}(t) = \frac{1}{t} \sum_{i=1}^{t} \mathbb{1}(\hat{y}_i = y_i)
     \]
  3. Update the model with \( (x_t, y_t) \) using:
     - `partial_fit` for ENFSUni0Evolving
     - warmup + `fit` blocks for the `eFS` models (ePL, ePL_plus, exTS,
       Simpl_eTS, eMG, ePL_KRLS_DISCO)

The script handles **numerical instabilities** gracefully:

- If a model produces `NaN` or `inf` predictions, a **fallback to a
  default class** is performed and a warning is printed.
- For ENFSUni0Evolving, the RLS update is equipped with **NaN
  protection** (resetting to a safe covariance if needed), and the
  script counts RLS instabilities.

## Outputs

For each combination `(feature_mode, drift_mode)` and each model:

- Prequential accuracy trajectory \(\text{Acc}(t)\)
- Final accuracy and mean accuracy over the stream
- Final number of rules
- For ENFSUni0Evolving:
  - number of RLS instabilities (`P_instabilities`)
  - number of prediction fallbacks, if any

The script generates:

- **Per-model plots**:
  - `acc_{model}_{feature_mode}.png` – accuracy over time.
  - `rules_{model}_{feature_mode}.png` – rule evolution (when available).

- **Multi-model plots**:
  - `acc_all_{dataset}_{feature_mode}_{drift_mode}.png` – overlay of
    accuracy curves for all models, with drift points marked.

- **LaTeX table**:
  - `apache_stream_evolving_summary.tex` – summarising final accuracy,
    mean accuracy and final #rules for all models and scenarios.

## How to run

From the `Pipeline/` folder:

```bash
(.venv) > python script3_stream_drift_fuzzy_baselines_apache.py


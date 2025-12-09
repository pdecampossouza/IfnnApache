
# Script 1 – Feature analysis for the Apache `due` dataset

This README documents the mathematical and methodological details of
`script1_feature_analysis_apache_due.py`.

## 1. Goal

Script 1 takes the **numeric** dataset produced by Script 0,
`apache_due_numeric.csv`, and performs a systematic feature analysis.
The main outputs are:

1. A table `feature_stats.csv` with basic statistics and correlations.
2. A JSON file `selected_features.json` with two ordered feature lists:
   - `features_fnn`  – up to 8 features recommended for the FNN model.
   - `features_enfs` – up to 20 features recommended for the eFNN model.
3. Two figures:
   - `feature_importance_rf.png` – RandomForest feature importances.
   - `feature_correlation_heatmap.png` – correlation heatmap for a subset.
4. Two LaTeX tables for direct inclusion in the thesis/paper:
   - `table_top10_features.tex` – top 10 features by RF importance.
   - `table_fnn_features.tex`  – statistics for the FNN feature set.

## 2. Notation

We reuse the notation from `README_Script0.md`. Let:

- \( X \in \mathbb{R}^{N \times d} \) be the matrix of candidate features
  (each column is one feature, each row is one issue).
- \( y \in \{0,1,2,3\}^N \) be the multi-class target `impact_class`.
- \( d_i \) be the `delaydays` value for issue \( i \).

Script 1 considers as **candidates** all numeric columns except:

- identifiers (`issuekey`),
- raw categorical columns (`type`, `priority`),
- the datetime (`openeddate`),
- explicit targets (`impact_class`, `y_binary`).

We also define a derived feature

\[
  \text{num\_deps} = \text{no\_issuelink} + \text{no\_blocking} + \text{no\_blockedby},
\]

interpreted as the total number of dependencies involving the issue.

## 3. Statistics and correlation

For each candidate feature \( x^{(j)} \) (column \( j \) of \( X \)), we
compute the following descriptive statistics:

- Mean: \( \mu_j = \frac{1}{N} \sum_{i=1}^N x_i^{(j)} \).
- Standard deviation: \( \sigma_j = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i^{(j)} - \mu_j)^2} \).
- Minimum and maximum values.

### 3.1. Pearson correlation with `delaydays`

To assess the linear relationship between each feature and the delay in
days, we compute the **Pearson correlation coefficient** between
feature \( x^{(j)} \) and the continuous variable \( d \):

\[
  \rho_j
  = \mathrm{corr}(x^{(j)}, d)
  = \frac{\sum_{i=1}^N (x_i^{(j)} - \bar{x}^{(j)})(d_i - \bar{d})}
         {\sqrt{\sum_{i=1}^N (x_i^{(j)} - \bar{x}^{(j)})^2}
          \sqrt{\sum_{i=1}^N (d_i - \bar{d})^2}}.
\]

This value is recorded in the column `corr_with_delaydays` of
`feature_stats.csv`.

## 3.2. RandomForest feature importance

To obtain a **supervised ranking** of features with respect to the
multi-class target `impact_class`, Script 1 fits a RandomForest
classifier:

\[
  f_{\text{RF}} : \mathbb{R}^d \to \{0,1,2,3\}.
\]

We use the scikit-learn implementation with the following hyperparameters:

- `n_estimators = 200`
- `class_weight = 'balanced_subsample'`
- `random_state = 42`

The RF is trained on a **standardised** version of the feature matrix:

\[
  z_i^{(j)} = \frac{x_i^{(j)} - \mu_j}{\sigma_j},
\]

to avoid scale effects. From the fitted forest, we extract the usual
**Gini importance** values:

\[
  I_j = \sum_{t \in T_j} \Delta G_t,
\]

where \( T_j \) is the set of splits based on feature \( j \) and
\( \Delta G_t \) is the reduction in Gini impurity at node \( t \).
These values are normalised by scikit-learn to sum to 1 across all
features.

The importances are stored in the column `rf_importance` of
`feature_stats.csv` and are also used to create the bar plot
`feature_importance_rf.png`.

## 4. Construction of feature sets for FNN and eFNN

Fuzzy models have an additional constraint: **human interpretability**.
In particular, we restrict the FNN to use at most 6–8 input variables
so that a human can still reason about the fuzzy rules.

Script 1 implements the following strategy:

1. Define a *curated pool* of interpretable features:
   - `RemainingDay`
   - `ProgressTime`
   - `perofdelay`
   - `workload`
   - `priority_level`
   - `no_comment`
   - `no_priority_change`
   - `num_deps`
   - `discussion`
   - `repetition`
   - `reporterrep`
   - `opened_year`
   - `opened_month`

2. Intersect this pool with the RF importance ranking, keeping the
   **order induced by importance**.

3. Define:

   - `features_fnn`: first 8 features from this ordered list.
   - `features_enfs`: start with the same ordered list, then append
     other features from the RF ranking until reaching 20 features
     in total.

The resulting lists are saved to `selected_features.json` and can be
imported directly by the modelling scripts.

## 5. LaTeX tables

Script 1 exports two LaTeX tables, stored in the feature-analysis
directory:

1. `table_top10_features.tex` – contains the 10 features with highest
   RF importance, along with their mean, standard deviation, min, max,
   correlation with `delaydays` and `rf_importance`.

2. `table_fnn_features.tex` – the same statistics, restricted to the
   features selected for the FNN model (`features_fnn`).

These tables can be included directly in LaTeX, for example:

```latex
\input{table_top10_features.tex}
```

or copied into larger table environments if desired.

## 6. Figures

- `feature_importance_rf.png`: bar plot of the top 20 features by RF importance.
- `feature_correlation_heatmap.png`: correlation matrix for the FNN feature set
  (or its first 15 variables).

## 7. How to run

From inside the `Pipeline` directory:

```bash
python script1_feature_analysis_apache_due.py
```

Requirements (in addition to Script 0):

- `scikit-learn`
- `seaborn`

The script assumes that Script 0 has already produced:

- `../apachedataset/processed/apache_due_numeric.csv`
- `../apachedataset/processed/apache_due_metadata.json`

and that there is write permission to create the subfolder
`../apachedataset/processed/feature_analysis/`.

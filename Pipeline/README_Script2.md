
# Script 2 – FNN vs. traditional classifiers on the Apache `due` dataset

This document explains the experimental design implemented in
`script2_fnn_vs_baselines_apache_due.py`.

## 1. Goal

Script 2 compares the **Fuzzy Neural Network (FNN)** model with several
traditional machine-learning classifiers on the Apache due-date dataset.
The comparison is performed on the **binary target**:

- `y_binary = 0` — issue with *no delay*,
- `y_binary = 1` — issue with *delay* (`delaydays > 0`).

The input features are those selected for the FNN in Script 1
(`features_fnn`), typically 6–8 interpretable dimensions.

The baselines are:

- RandomForest classifier;
- Feed-forward neural network (MLP);
- Gaussian Naive Bayes;
- Support Vector Machine (RBF kernel).

All models are evaluated on the same train/test splits.

## 2. Data and experimental protocol

Let:

- \( N \) be the total number of issues;
- \( d \) be the number of selected features (from Script 1);
- \( X \in \mathbb{R}^{N \times d} \) the feature matrix;
- \( y \in \{0,1\}^N \) the binary target (`y_binary`).    

For each run \( r = 1, \dots, R \) (with \( R = 10 \)):

1. The data are split into training and testing sets using a **stratified**
   split with proportion:

   \[
     70\\% \\text{ training}, \\quad 30\\% \\text{ testing}.
   \]

   The scikit-learn function `train_test_split` is used with a different
   random seed in each run, but the **same split** is used for all models
   within that run.

2. The FNN is trained on the **raw features** \( X \) (no scaling), to keep
   the interpretability of membership functions in the original space.

3. Baseline models are trained on **standardised features**, obtained by:

   \[
     z_i^{(j)} = \\frac{x_i^{(j)} - \\mu_j}{\\sigma_j},
   \]

   where \( \\mu_j \) and \( \\sigma_j \) are the mean and standard
   deviation of feature \( j \) computed on the training set only.

The number of runs \( R = 10 \) is chosen so that the effect of random
train/test splits and random initialisations is smoothed out by averaging.

## 3. FNN model (binary setting)

The FNN implementation comes from `models.models.FNNModel`. It expects
labels in the set \\( \\{-1, +1\\} \\). Script 2 converts the binary target via

\\[
  y_i' =
  \\begin{cases}
    +1, & y_i = 1, \\\\
    -1, & y_i = 0.
  \\end{cases}
\\]

During training, the FNN computes fuzzy memberships, logic neuron outputs
and finally optimises a linear output layer using the Moore–Penrose
pseudo-inverse. The prediction for a test sample is based on the sign of
the final output:

\\[
  \\hat{y}_i' = \\mathrm{sign}(f_{\\text{FNN}}(x_i)), \\quad
  \\hat{y}_i = \\mathbb{I}\\{ \\hat{y}_i' > 0 \\},
\\]

where \\( \\hat{y}_i \\in \\{0,1\\} \\) is the predicted class in the
original binary space. Script 2 then computes the same metrics as for the
baseline models.

The FNN uses:

- `num_mfs = 3` membership functions per input dimension;
- neuron type `"andneuron"`;
- activation `"linear"`;
- optimizer `"moore-penrose"`;
- no pruning (`"none"`).

These choices can be adjusted directly in the configuration section of the
script.

## 4. Evaluation metrics

For each model \( m \) and each run \( r \), Script 2 computes:

- Accuracy:

  \\[
    \\mathrm{Acc}_{m,r} = \\frac{TP + TN}{TP + TN + FP + FN},
  \\]

- Precision:

  \\[
    \\mathrm{Prec}_{m,r} = \\frac{TP}{TP + FP},
  \\]

- Recall:

  \\[
    \\mathrm{Rec}_{m,r} = \\frac{TP}{TP + FN},
  \\]

- F1-score:

  \\[
    \\mathrm{F1}_{m,r} = \\frac{2 \\cdot \\mathrm{Prec}_{m,r} \\cdot \\mathrm{Rec}_{m,r}}%
                              {\\mathrm{Prec}_{m,r} + \\mathrm{Rec}_{m,r}}.
  \\]

where TP, TN, FP, FN are the usual entries of the confusion matrix with
respect to the positive class (delayed issues, `y_binary = 1`).

The implementation uses `sklearn.metrics.accuracy_score` and
`precision_recall_fscore_support` with `average="binary"`.

## 5. Aggregation over runs

Let \\( R = 10 \\) be the number of runs. For each model \\( m \\) and
metric \\( \\mu \\in \\{\\mathrm{Acc}, \\mathrm{Prec}, \\mathrm{Rec}, \\mathrm{F1}\\} \\),
Script 2 computes:

- Mean value:

  \\[
    \\bar{\\mu}_m = \\frac{1}{R} \\sum_{r=1}^R \\mu_{m,r},
  \\]

- Sample standard deviation:

  \\[
    s_m = \\sqrt{ \\frac{1}{R-1} \\sum_{r=1}^R (\\mu_{m,r} - \\bar{\\mu}_m)^2 }.
  \\]

The results are stored in `summary_metrics.csv` and also exported as a
LaTeX table `results_table.tex`.

## 6. Figures

Two bar plots are generated:

1. `model_comparison_accuracy.png` – mean accuracy with error bars
   corresponding to one standard deviation.

2. `model_comparison_f1.png` – same for the F1-score.

These figures can be directly used in the thesis to illustrate the
comparative performance between FNN and traditional models.

## 7. Fuzzy rules and interpretability

For the run with seed 0, Script 2 stores the fuzzy rules learned by the
FNN:

- `fnn_rules_seed0.txt` – full list of rules in plain text.
- `fnn_rules_seed0.tex` – LaTeX table containing a subset (e.g., the
  first 15 rules) in a single-column format.

These rules have the general structure

```text
IF x1 is MF1 AND x2 is MF2 AND ... THEN output is w
```

where the consequent weight \\( w \\) encodes the impact on the final
decision. The mapping between linguistic terms (MF1, MF2, etc.) and the
original features is determined by the configuration of membership
functions inside the FNN.

In the methodology chapter, you can reference this script as the
implementation of the comparative experiment and rule extraction
(e.g., "see Script 2 for details on the experimental protocol and
evaluation").

## 8. How to run

From inside the `Pipeline` directory:

```bash
python script2_fnn_vs_baselines_apache_due.py
```

Requirements:

- `numpy`, `pandas`, `matplotlib`, `tqdm`
- `scikit-learn`
- The `models` package containing `FNNModel` must be available one level
  above the `Pipeline` directory (as in the current folder structure).

The script assumes that Scripts 0 and 1 have already produced:

- `../apachedataset/processed/apache_due_numeric.csv`
- `../apachedataset/processed/feature_analysis/selected_features.json`

and will write all results under:

- `../apachedataset/processed/feature_analysis/fnn_vs_baselines/`.

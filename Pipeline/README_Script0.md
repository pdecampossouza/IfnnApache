
# Script 0 – Preprocessing of the Apache `due` dataset

This document explains, in a technical and mathematical way, what
`script0_preprocess_apache_due.py` does.

## 1. Goal

Script 0 takes the **raw** `apache_due.csv` file from the JIRA-Estimation-Prediction
repository and produces a **fully numeric** dataset that can be reused by all
subsequent experiments (feature analysis, FNN, eFNN).

The script also computes target variables compatible with the study of
Choetkiertikul et al. (2017):

- a **4-class impact variable** `impact_class`;
- a **binary** delay indicator `y_binary`.

It finally saves:

1. `apache_due_numeric.csv` – numeric table.
2. `apache_due_metadata.json` – metadata (mappings, thresholds, etc.).
3. Two PNG figures with basic histograms of the delay distribution.

## 2. Notation

Let

- \( N \) be the number of issues (rows) in the dataset.
- For each issue \( i \in \{1, \dots, N\} \):
  - \( d_i \) = `delaydays` (delay in days, real-valued).
  - `type_i`  ∈ {Bug, Task, NewFeature, ...} – textual type label.
  - `priority_i` ∈ {Trivial, Minor, Major, Critical, ...} – textual priority.

The goal of Script 0 is to construct numeric variables suitable for machine
learning models:

- Encoded type \( t_i \in \mathbb{N} \).
- Encoded priority \( p_i \in \mathbb{N} \).
- Impact class \( c_i \in \{0,1,2,3\} \).
- Binary delay indicator \( y_i \in \{0,1\} \).

## 3. Transformations

### 3.1. Impact classes from `delaydays`

We denote the delay in days by \( d_i \). Script 0 computes the empirical
distribution of **positive delays**:

\\[
  D^+ = \\{ d_i : d_i > 0 \\}.
\\]

Then it computes two empirical percentiles:

\\[
  T_1 = \\mathrm{percentile}_{p_\\text{low}}(D^+), \\quad
  T_2 = \\mathrm{percentile}_{p_\\text{high}}(D^+),
\\]

where by default \( p_\\text{low} = 33.3 \\) and \( p_\\text{high} = 66.6 \\).

Using these thresholds we define the **impact class** \( c_i \) as

\\[
  c_i =
  \\begin{cases}
    0, & d_i = 0         & \\text{(no delay)},\\\\
    1, & 0 < d_i \\le T_1 & \\text{(small delay)},\\\\
    2, & T_1 < d_i \\le T_2 & \\text{(medium delay)},\\\\
    3, & d_i > T_2       & \\text{(large delay)}.
  \\end{cases}
\\]

This is the multi-class target used later to reproduce the classification
scenario of Choetkiertikul et al. (2017).

The thresholds \( T_1, T_2, p_\\text{low}, p_\\text{high} \) are stored in
`apache_due_metadata.json` under the key `delaydays_thresholds`.

### 3.2. Binary delay indicator

The binary indicator is defined simply as

\\[
  y_i =
  \\begin{cases}
    0, & d_i = 0,\\\\
    1, & d_i > 0.
  \\end{cases}
\\]

This variable is called `y_binary` in the numeric CSV and metadata.

### 3.3. Encoding categorical variables

Let \( \\mathcal{T} \) be the set of distinct textual values of the column
`type`, and \( \\mathcal{P} \) the set of distinct values of `priority` in the
dataset. Script 0 builds **deterministic mappings**:

\\[
  \\phi_T : \\mathcal{T} \\to \\{1, \\dots, |\\mathcal{T}|\\}, \\quad
  \\phi_P : \\mathcal{P} \\to \\{1, \\dots, |\\mathcal{P}|\\},
\\]

by sorting the categories alphabetically and assigning consecutive integers
starting at 1.

For each issue \( i \):

\\[
  t_i = \\phi_T(\\text{type}_i), \\quad
  p_i = \\phi_P(\\text{priority}_i).
\\]

These numeric codes are stored in the columns

- `type_code` for \( t_i \);
- `priority_level` for \( p_i \).

The mappings \( \\phi_T, \\phi_P \\) are saved in the JSON metadata as
`type_mapping` and `priority_mapping`, and can be used later to interpret
fuzzy rules (e.g., mapping integer codes back to human-readable labels).

### 3.4. Date handling

The column `openeddate` is converted to a pandas `datetime` object. Script 0
additionally derives:

- `opened_year`  = calendar year of the opening date.
- `opened_month` = calendar month.

These two variables are optional features for later temporal analysis (e.g.,
concept drift over time) and are kept in the numeric CSV.

## 4. Figures

Script 0 generates two histograms using `matplotlib`:

1. `delaydays_histogram.png` – empirical distribution of `delaydays`.
2. `impact_class_histogram.png` – empirical distribution of `impact_class`.

These figures are intended to be included in the thesis/paper to illustrate
the delay structure for the Apache project.

## 5. How to run

From inside the `Pipeline` directory:

```bash
python script0_preprocess_apache_due.py
```

Requirements:

- Python 3.8+
- `pandas`, `numpy`, `tqdm`, `matplotlib`

The script assumes that:

- `../apachedataset/apache_due.csv` exists.
- You have write permission to create `../apachedataset/processed/`.

## 6. Relation to later scripts

- **Script 1** (feature analysis) will *only* use the numeric CSV and
  metadata produced here.
- For a deeper understanding of the construction of the targets and
  encodings inside Script 1 or the modelling scripts, refer back to:
  - Section 3.1 (impact classes);
  - Section 3.3 (categorical encodings) of this README.

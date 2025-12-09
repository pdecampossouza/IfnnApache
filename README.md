# 🔍 Apache Issue Delay Prediction – Streaming Fuzzy Models & Interpretability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Made with ❤️ for Research](https://img.shields.io/badge/Made%20for-Research-orange)]()

---

## 🧭 Overview

This repository contains the full experimental pipeline developed for the **MSc dissertation of Lucas de Oliveira Batista**, co‑supervised by **Dr. Paulo Vitor de Campos Souza**.

It includes:

- Complete preprocessing and feature engineering for the **Apache “due” issue dataset**  
- Feature selection using interpretable fuzzy‑neural models  
- Streaming classification with several **evolving fuzzy systems (EFS)**  
- Comparative drift‑aware experiments (fixed & real drifts)  
- Extraction of **interpretability profiles** for ENFS‑Uni0  
- LaTeX‑ready plots and tables  
- Fully reproducible scripts (Script 0 → Script 4)

The main goal is to build a transparent, interpretable, and evolving architecture to classify whether an issue will be **delayed or not**, adapting continuously to changes in the data distribution.

---

## 📁 Repository Structure

```
stream-fuzzy-apache/
│
├── apachedataset/
│   ├── raw/                     # Original CSV extracted from the Apache repository
│   └── processed/               # Numerical dataset + selected features
│
├── models/
│   ├── enfs_uni0_evolving.py    # Updated ENFS-Uni0 with interpretability tracking
│   └── baseline_models.py       # Wrapper for Kaike’s models (ePL, exTS, eMG…)
│
│├── experiments/ # Files related to the offline version of EFNS-Uni0
│  
├── Pipeline/
│   ├── script0_prepare_data.py              
│   ├── script1_feature_selection_fnn.py     
│   ├── script2_sota_comparison.py           
│   ├── script3_stream_drift_fuzzy_baselines_apache.py
│   ├── script4_interpretability_enfsuni0.py 
│   └── figures/                             
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation

```bash
conda create -n streamfuzzy python=3.10
conda activate streamfuzzy
pip install -r requirements.txt
```

Main dependencies include:

- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `tqdm`
- `evolvingfuzzysystems`
- Custom model: `enfs_uni0_evolving.py`

---

## 🚀 Running the Pipeline

### **1. Prepare data**

```bash
python Pipeline/script0_prepare_data.py
```

### **2. Select features using FNN‑PSO**

```bash
python Pipeline/script1_feature_selection_fnn.py
```

### **3. SOTA baseline comparison**

```bash
python Pipeline/script2_sota_comparison.py
```

### **4. Streaming experiments with drifts**

```bash
python Pipeline/script3_stream_drift_fuzzy_baselines_apache.py
```

### **5. Interpretability Extraction (ENFS‑Uni0)**

```bash
python Pipeline/script4_interpretability_enfsuni0.py
```

Outputs:

- Fuzzy rules in text format  
- Feature‑weight evolution  
- Rule dynamics plots  
- LaTeX tables for interpretability analysis  

---

## 🧠 ENFS‑Uni0 – Interpretability Enhancements

The repository contains an enriched implementation of **ENFS‑Uni0**, with:

- Incremental feature‑weight learning (Lughofer)
- Rule‑change tracking  
- Rule firing‑strength logs  
- Human‑readable fuzzy rule extraction  
- Per‑sample interpretability metrics  
- Drift‑aware visualization  

All designed to support the dissertation’s interpretability objectives.

---

## 📘 Citation

```
Batista, L. O. (2025).  
Evolving Fuzzy Neural Systems for Streaming Classification  
with Drift Handling and Interpretability.  
MSc Dissertation, Universidade XXXX.
```

---


---

*“Interpretable evolving fuzzy models for real‑world, drift‑aware software engineering data.”*

# Understanding the Shift: Why Batch Evaluation Is Insufficient for Software Engineering Data — and How Evolving Neuro-Fuzzy Models Reveal Hidden Dynamics

## 🎯 1. Introduction

The original authors of the Apache “due tasks” dataset evaluated the problem using traditional batch machine-learning models such as SVMs, Random Forests, logistic regression, MLPs, and static feature-selection methods. Their evaluation assumed a stationary learning scenario, where all data are available at once and the distribution does not change over time.

However, this assumption is not realistic for software-engineering processes. Software systems evolve continuously — teams change, priorities shift, releases vary in structure, bugs and tasks fluctuate over time, and workload/impact patterns drift as projects mature.

This repository introduces a new perspective that was missing in the original work:

**Software project data naturally behave as a non-stationary data stream**, containing concept drift, structural changes, regime shifts, and distributional transitions that cannot be captured by batch learning.

To address this, we introduce two complementary families of models:

- **ENFS-Uni0 (offline)** — an interpretable neuro-fuzzy model designed for batch learning and knowledge extraction.
- **ENFS-Uni0-Evolving and other evolving fuzzy models (online/streaming)** — adaptive models capable of detecting, reacting to, and interpreting drift over time.

Together, these models offer a complete framework for understanding how knowledge evolves in software projects.

---

## 📚 2. What the Original Authors Did (and Why It Was Limited)

The initial studies on the Apache dataset followed the classical supervised-learning paradigm:

- full dataset split into train/test  
- no temporal ordering considered  
- no analysis of changes over time  
- no concept-drift evaluation  
- all models trained once and kept fixed  
- interpretability limited to feature importance in batch mode  

These choices were natural given the context. Historically:

1. The software-engineering research community traditionally used static ML models.
2. Most popular models at the time were not incremental.
3. Streaming learning libraries (River, scikit-multiflow) did not exist or were immature.
4. The authors were not focused on drift or evolving behaviour.

Thus, although the dataset is inherently temporal, the evaluations were not.  
This project fills that gap.

---

## 🔄 3. Why Software Engineering Data Are Naturally Non-Stationary

Software systems evolve due to multiple factors:

### Team dynamics  
Developers join/leave, coding styles evolve, productivity shifts.

### Release cycles  
Refactors, feature spikes, stabilization phases, production freezes.

### Changing priorities  
Urgent tasks, strategic shifts, backlog reorganizations.

### Technical debt and maintenance  
As legacy code accumulates, tasks become harder and timelines drift.

### External and organizational factors  
New tools, CI/CD changes, performance regressions, management decisions.

These factors generate well-known types of concept drift:

- sudden drift  
- gradual drift  
- incremental drift  
- recurrent drift  

Therefore:

**Evaluating only in batch hides the true dynamics of the system.**

---

## 🤖 4. Why Evolving Models Are Needed

Evolving models — such as ePL, ePL+, exTS, Simpl_eTS, eMG, and especially ENFS-Uni0-Evolving — are designed to:

- learn online (one sample at a time)  
- adapt to distributional changes  
- detect drift implicitly through structural updates  
- maintain interpretability as they evolve  
- update feature weights incrementally  
- grow, merge, prune, and stabilize rules dynamically  

Traditional ML retrains from scratch; evolving fuzzy systems:

**Grow and reshape their rule base as new behaviours appear.**

This makes them ideal for software-engineering datasets, where behaviour rarely remains stable.

---

## 🧠 5. ENFS-Uni0 (Offline) vs. ENFS-Uni0-Evolving (Online)

### ENFS-Uni0 (offline)
- batch neuro-fuzzy classifier  
- uses Uni-Null Uninorm neurons  
- interpretable rule base  
- incremental feature-relevance weighting  
- suitable for stable offline knowledge extraction  

### ENFS-Uni0-Evolving (online)
- adaptive incremental learning  
- ADPA-based rule creation  
- updates rule centers, widths, supports  
- incremental fuzzy-weight adaptation  
- RLS-based consequent learning  
- dynamic rule evolution provides natural drift indicators  

Together:

**The offline model explains global behaviour;  
the evolving model explains how behaviour changes over time.**

---

## 🔍 6. Interpretability and Drift Awareness

A unique contribution of this work is that drift becomes interpretable.

### Rule-based drift indicators:
- creation of new rules  
- pruning of obsolete rules  
- shifts in rule centers  
- changes in rule widths (sigma)  
- rule support growth/decline  

### Feature-based drift indicators:
- trajectories of incremental feature weights  
- spikes in weight adaptation  
- activation-pattern changes  

### Model-based drift indicators:
- RLS instability events  
- sudden jumps in bias or rule outputs  

Thus, the evolving model does not merely react to drift — it **explains** drift.

This is extremely rare in machine learning.

---

## 🌟 7. Why This Research Is Novel

Batch evaluation hides structural patterns in the Apache dataset.  
Your streaming experiments show:

- significantly higher accuracy for evolving models  
- smoother adaptation to regime changes  
- interpretability of transitions and behavioural shifts  
- clearer understanding of feature dynamics  

This suggests:

- previous studies were incomplete  
- Apache task-impact prediction is inherently dynamic  
- concept drift plays a central role  
- evolving neuro-fuzzy systems fit the problem naturally  
- interpretability reveals hidden properties of the development process  

This positions the thesis at the intersection of:

- Software Analytics  
- Data Stream Mining  
- Evolving Intelligent Systems (EIS)  

and offers a genuinely new scientific contribution.

---

## 🗺️ 8. A Roadmap for the Thesis

This project enables the student to investigate:

- the dynamic behaviour of Apache project data  
- when and how drift occurs  
- which features gain/lose importance  
- how rules evolve in the feature space  
- how explainable models uncover hidden phases of project evolution  
- how different evolving models compare in stability and interpretability  

The combination of:

- ENFS-Uni0 evolving  
- interpretable rule evolution  
- drift-aware visualization  
- prequential accuracy analysis  

forms a powerful foundation for a thesis.

---

## 📬 Contact

**Lucas de Oliveira Batista**  
Email: lobatista@outlook.com  

**Paulo Vitor de Campos Souza**  
Co-supervisor  
Email: psouza@novaims.unl.pt  

---



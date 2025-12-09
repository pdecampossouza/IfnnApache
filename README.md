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

## 📬 Contact

**Lucas de Oliveira Batista**  
Email: lobatista@outlook.com  

**Dr. Paulo Vitor de Campos Souza**  
Email: psouza@novaims.unl.pt  

---

*“Interpretable evolving fuzzy models for real‑world, drift‑aware software engineering data.”*

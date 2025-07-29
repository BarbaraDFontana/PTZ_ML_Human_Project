# PTZ_ML_Human_Project

This repository contains all datasets, scripts, and resources used in the manuscript:

**"Human Annotation vs. Machine Learning Models: Optimizing Zebrafish Behavioral Classification for Seizure Analysis"**  
*Fontana et al., 2025 – In preparation*

---

## 📄 Project Overview

We benchmark five supervised machine learning (ML) models (Random Forest, XGBoost, SVM, kNN, and MLP) for classifying seizure-like behaviors in adult zebrafish. Using over **43,000 human-labeled video frames**, we directly compare ML outputs to **expert consensus**, analyze **inter-rater variability**, and evaluate the effects of **post-processing and temporal resolution**.

---
## 🔬 Main Experiments

1. **Human annotation analysis**
   - Annotation time, behavioral complexity
   - Inter-rater agreement (Fleiss' κ)
   - Learning/fatigue effects

2. **ML model training & benchmarking**
   - Five classifiers trained on human-labeled data
   - 10-fold cross-validation, internal and external validation

3. **Post-processing & temporal resolution**
   - Accuracy changes with speed/posture filters
   - Comparison of frame-by-frame vs. 5s/15s/30s scoring

---

## 🔗 Data and Reproducibility

- All datasets and trained model outputs are available in this repository.
- Scripts are written in **R (v4.4)** 

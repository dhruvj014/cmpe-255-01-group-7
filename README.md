# CMPE 255-01 — Data Mining Project  
## Group 7

### Team Members (Alphabetical Order)
- **Dhruv Sachin Jain** (SJSU ID: 019150859)  
- **Disha Jadav** (SJSU ID: 018484362)  
- **Himanshu Jain** (SJSU ID: 019098794)  
- **Nitish Kumar** (SJSU ID: 019155916)

## Project Overview

**Fake Review Detection on Yelp Using Multi-Signal Analysis**

This project investigates the detection of fake reviews on Yelp using a **multi-signal data mining framework** applied to the **YelpZip dataset (608,598 reviews)**. The goal is to determine whether combining multiple signals—**textual content, reviewer behavior patterns, association rules, clustering structures, and anomaly detection signals**—can outperform traditional single-signal fake review detection approaches.

The system is designed as a **six-layer pipeline** that includes data preprocessing and OLAP analysis, association rule mining (FP-Growth), transformer-based text classification using **DeBERTa-v3**, reviewer behavior clustering (K-Means and DBSCAN), supervised classification models (Decision Tree, Random Forest, SVM, MLP), and unsupervised anomaly detection methods (Isolation Forest and LOF).

A comprehensive **ablation study** evaluates the contribution of each signal type by progressively combining features from text-only models to a full multi-signal model. Performance is measured using **F1-score, AUC-ROC, and Precision@K**, with additional validation through **synthetic attack injection and cross-domain testing** on the Ott Deceptive Opinion Spam dataset. The project aims to demonstrate that integrating linguistic and behavioral signals leads to **more robust and accurate fake review detection systems**.
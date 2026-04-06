# CMPE 255-01: Data Mining Project  
## Group 7

### Team Members
- **Dhruv Sachin Jain** (SJSU ID: 019150859)  
- **Disha Jadav** (SJSU ID: 018484362)  
- **Himanshu Jain** (SJSU ID: 019098794)  
- **Nitish Kumar** (SJSU ID: 019155916)

## Project Overview

**Fake Review Detection on Yelp Using Multi-Signal Analysis**

This project investigates the detection of fake reviews on Yelp using a **multi-signal data mining framework** applied to the **YelpZip dataset (608,598 reviews)**. The goal is to determine whether combining multiple signals—**textual content, reviewer behavior patterns, association rules, clustering structures, and anomaly detection signals**—can outperform traditional single-signal fake review detection approaches.

The system is designed as a **six-layer pipeline** that includes data preprocessing and OLAP analysis, association rule mining (FP-Growth), transformer-based text classification using **DeBERTa-v3**, reviewer behavior clustering (K-Means and DBSCAN), supervised classification models (Decision Tree, Random Forest, SVM, MLP), and unsupervised anomaly detection methods (Isolation Forest and LOF).

A comprehensive **ablation study** evaluates the contribution of each signal type by progressively combining features from text-only models to a full multi-signal model. Performance is measured using **F1-score, AUC-ROC, and Precision@K**, with additional validation through **synthetic attack injection and cross-domain testing** on the Ott Deceptive Opinion Spam dataset. The project aims to demonstrate that integrating linguistic and behavioral signals leads to **more robust and accurate fake review detection systems**.

---

## Layer 2: FP-Growth Association Rule Mining

FP-Growth discovers frequent co-occurring behavioral traits among reviewers and links them to spam rates. It was chosen over Apriori because it avoids expensive candidate generation, making it tractable on 260,277 reviewer profiles.

**Input:** `reviewer_features.csv` (produced by L1)

**Outputs:**
- `L2_FPGrowth/outputs/encoded_baskets.csv` — reviewer behavioral baskets (discretized)
- `L2_FPGrowth/outputs/frequent_itemsets.csv` — all frequent itemsets at min_support=0.05
- `L2_FPGrowth/outputs/association_rules.csv` — full rule set with support, confidence, lift
- `L2_FPGrowth/outputs/spam_correlated_rules.csv` — rules filtered to antecedent_spam_rate > 20%

**Key Finding:** The strongest spam-correlated rule is `{tenure=new, review_count=Low}` → `{burst=Normal, seller_conc=High}` with 100% confidence (lift 1.24); reviewers matching these antecedents have a 29.1% spam rate — 2.2× the dataset average of 13.2%.

**How to run:**
```bash
pip install mlxtend
python L2_FPGrowth/01_basket_encoding.py
python L2_FPGrowth/02_fpgrowth_mining.py
python L2_FPGrowth/03_rule_analysis.py
```
Or open `03_fpgrowth_association_rules.ipynb`

---

## Layer 4: K-Means + DBSCAN Reviewer Clustering

Reviewers are clustered by behavioral features (without labels) to find groups with elevated spam rates and extreme outlier accounts. K-Means provides interpretable broad groupings; DBSCAN surfaces coordinated spammer rings as density outliers.

**Input:** `reviewer_features.csv` (produced by L1)

**Outputs:**
- `L4_Clustering/outputs/reviewer_clusters.csv` — K-Means cluster assignment per reviewer
- `L4_Clustering/outputs/dbscan_results.csv` — DBSCAN cluster + noise assignment per reviewer
- `L4_Clustering/outputs/cluster_spam_summary.csv` — spam rate and profile per cluster (both methods)

**Key Finding:** K-Means Cluster 3 (40,914 reviewers): 36.5% spam rate, profile: new accounts + uniform ratings — 2.8× the dataset average. DBSCAN Cluster 13 (19,051 reviewers): 47.7% spam rate (3.6× average); micro-cluster 48 (90 reviewers) reaches 52.8% spam rate (4.0× average), indicating tightly coordinated spammer rings.

**How to run:**
```bash
python L4_Clustering/01_preprocessing.py
python L4_Clustering/02_kmeans_clustering.py
python L4_Clustering/03_dbscan_clustering.py
python L4_Clustering/04_cluster_analysis.py
```
Or open `04_clustering.ipynb`

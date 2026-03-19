# L1 ETL/OLAP Pipeline - YelpZip Fake Review Detection

**Project:** Fake Review Detection on Yelp Using Multi-Signal Analysis
**Layer:** L1 (Data Engineering Foundation)
**Owner:** Himanshu Jain | Group 7 - CMPE 255

## Overview

This pipeline processes the YelpZip dataset (~608K reviews) through a complete ETL workflow:

1. **Extract**: Load raw data with robust parsing for edge cases
2. **Transform**: Engineer features at review, reviewer, and seller levels
3. **Load**: Store as Delta Lake tables and CSV exports
4. **OLAP**: Build multidimensional cubes for analysis
5. **Visualize**: Generate publication-quality plots

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Java 8+ is installed for Spark
java -version
```

## Usage

```bash
# Run the complete pipeline
python main.py
```

The pipeline will:
- Process all 608,598 reviews
- Generate reviewer profiles for 260,277 users
- Generate seller profiles for 5,044 businesses
- Build 4 OLAP cubes
- Create 6 visualization plots
- Run 8 quality validation checks

## Output Structure

```
L1_ETL_OLAP/
├── delta_lake/
│   ├── reviews_enriched/      # 608,598 rows, partitioned by year_month
│   ├── reviewer_profiles/     # 260,277 rows
│   ├── seller_profiles/       # 5,044 rows
│   └── olap_cubes/
│       ├── time_x_rating/
│       ├── tenure_x_rating/
│       ├── time_x_tenure/
│       └── full_3d/
├── output_csv/
│   ├── reviews_enriched.csv
│   ├── reviewer_profiles.csv
│   ├── seller_profiles.csv
│   ├── cube_time_x_rating.csv
│   ├── cube_tenure_x_rating.csv
│   ├── cube_time_x_tenure.csv
│   └── cube_full_3d.csv
└── plots/
    ├── heatmap_month_x_rating.png
    ├── heatmap_tenure_x_rating.png
    ├── line_monthly_spam_rate.png
    ├── bar_spam_by_tenure.png
    ├── grouped_bar_rating_dist.png
    └── reviewer_feature_distributions.png
```

## Output Schemas

### reviews_enriched

| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Numeric reviewer ID |
| prod_id | int | Numeric product/business ID |
| rating | float | Star rating (1.0-5.0) |
| label | int | Original label (-1=spam, +1=legitimate) |
| date | string | Review date (YYYY-MM-DD) |
| review_text | string | Full review text (140 may be null) |
| business_name | string | Business name |
| is_spam | int | Binary spam flag (1=spam, 0=legitimate) |
| review_date | date | Parsed date |
| year | int | Year extracted |
| month | int | Month extracted |
| week_number | int | Week of year |
| day_of_week | int | Day of week (1=Sunday) |
| year_month | string | Year-month (YYYY-MM) |
| review_length | int | Character count |
| word_count | int | Word count |
| exclamation_count | int | Number of '!' |
| question_count | int | Number of '?' |
| capital_ratio | double | Uppercase ratio |
| avg_word_length | double | Average word length |

### reviewer_profiles

| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Numeric reviewer ID |
| review_count | int | Total reviews written |
| spam_count | int | Number of spam reviews |
| spam_rate | double | Fraction of spam reviews |
| avg_rating | double | Average rating given |
| rating_std | double | Rating standard deviation |
| avg_review_length | double | Average character count |
| avg_word_count | double | Average word count |
| first_review_date | date | Earliest review |
| last_review_date | date | Latest review |
| unique_sellers | int | Distinct businesses reviewed |
| tenure_days | int | Days between first and last review |
| reviews_per_week | double | Average weekly review rate |
| max_seller_fraction | double | Concentration on single seller |
| avg_days_between_reviews | double | Average gap between reviews |
| burst_score | int | Max reviews in 7-day window |
| rating_entropy | double | Shannon entropy of ratings |

### seller_profiles

| Column | Type | Description |
|--------|------|-------------|
| prod_id | int | Numeric product/business ID |
| business_name | string | Business name |
| total_reviews | int | Total reviews received |
| spam_reviews | int | Number of spam reviews |
| spam_rate | double | Fraction of spam reviews |
| unique_reviewers | int | Distinct reviewers |
| avg_rating | double | Average rating received |
| rating_std | double | Rating standard deviation |
| active_days | int | Days between first and last review |
| review_velocity | double | Reviews per week |
| suspicious_reviewer_fraction | double | Fraction of reviewers with >50% spam |

## Teammate Handoff

### Dhruv - L2 (FP-Growth Association Rules)
- **Input**: `reviewer_profiles.csv` + `reviews_enriched.csv`
- Discretize features into basket items for association mining

### Nitish - L3 (DeBERTa Fine-tuning)
- **Input**: `reviews_enriched.csv`
- Text column: `review_text`, Label: `is_spam`
- Handle 13.2% class imbalance with stratified split

### Dhruv - L4 (K-Means / DBSCAN Clustering)
- **Input**: `reviewer_profiles.csv`
- Apply StandardScaler to all behavioral features
- Consider log-transform for skewed features

### Disha - L5 (Classification / Anomaly Detection)
- **Input**: `reviewer_profiles.csv` or `reviews_enriched.csv`
- Use all features for classification
- Apply group-aware split by user_id to prevent leakage

## Data Quality Notes

- 140 reviews have null `review_text` due to embedded tab characters
- 13.2% spam rate (80,466 spam / 608,598 total)
- Date range: 2004-10-20 to 2015-01-10
- 65% of reviewers have only 1 review (single-review edge cases handled)

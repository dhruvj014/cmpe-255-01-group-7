#!/usr/bin/env python3
"""Main entry point for YelpZip L1 ETL/OLAP Pipeline.

This pipeline processes the YelpZip fake review dataset through:
1. Data extraction with robust parsing
2. Review-level feature engineering
3. Reviewer-level aggregation
4. Seller-level aggregation
5. OLAP cube construction
6. Visualization generation
7. Quality validation

Author: Himanshu Jain | Group 7 - CMPE 255
"""
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F

from utils.spark_session import create_spark_session, stop_spark_session
from etl.extract import extract_all_data, EXPECTED_METADATA_ROWS, EXPECTED_REVIEWER_COUNT, EXPECTED_PRODUCT_COUNT
from etl.transform_reviews import transform_reviews
from etl.transform_reviewers import transform_reviewers
from etl.transform_sellers import transform_sellers
from etl.load import save_all_tables, save_olap_cubes
from olap.cube_builder import build_olap_cubes
from olap.visualizations import generate_all_visualizations


# Configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "YelpZip")
DELTA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta_lake")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_csv")
PLOTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def print_header(step_num: int, total_steps: int, title: str) -> None:
    """Print a formatted step header."""
    print("\n" + "=" * 60)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 60)


def run_quality_checks(reviews_df, reviewer_df, seller_df) -> bool:
    """Run quality validation checks from Section J of the spec.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles DataFrame.
        seller_df: Seller profiles DataFrame.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    all_passed = True

    # Check 1: Row counts
    print("\n[1] Row count checks...")
    reviews_count = reviews_df.count()
    if reviews_count == EXPECTED_METADATA_ROWS:
        print(f"    PASS: Reviews = {reviews_count:,} (expected {EXPECTED_METADATA_ROWS:,})")
    else:
        print(f"    FAIL: Reviews = {reviews_count:,} (expected {EXPECTED_METADATA_ROWS:,})")
        all_passed = False

    reviewer_count = reviewer_df.count()
    if reviewer_count == EXPECTED_REVIEWER_COUNT:
        print(f"    PASS: Reviewers = {reviewer_count:,} (expected {EXPECTED_REVIEWER_COUNT:,})")
    else:
        print(f"    FAIL: Reviewers = {reviewer_count:,} (expected {EXPECTED_REVIEWER_COUNT:,})")
        all_passed = False

    seller_count = seller_df.count()
    if seller_count == EXPECTED_PRODUCT_COUNT:
        print(f"    PASS: Sellers = {seller_count:,} (expected {EXPECTED_PRODUCT_COUNT:,})")
    else:
        print(f"    FAIL: Sellers = {seller_count:,} (expected {EXPECTED_PRODUCT_COUNT:,})")
        all_passed = False

    # Check 2: Spam rate
    print("\n[2] Spam rate check...")
    actual_spam_rate = reviews_df.filter(F.col("is_spam") == 1).count() / reviews_df.count()
    expected_spam_rate = 0.132
    if abs(actual_spam_rate - expected_spam_rate) < 0.005:
        print(f"    PASS: Spam rate = {actual_spam_rate:.3f} (expected ~{expected_spam_rate})")
    else:
        print(f"    FAIL: Spam rate = {actual_spam_rate:.3f} (expected ~{expected_spam_rate})")
        all_passed = False

    # Check 3: No nulls in key columns
    print("\n[3] Null checks in key columns...")
    key_columns = ["user_id", "prod_id", "rating", "label", "review_date", "is_spam"]
    for col in key_columns:
        null_count = reviews_df.filter(F.col(col).isNull()).count()
        if null_count == 0:
            print(f"    PASS: No nulls in {col}")
        else:
            print(f"    FAIL: {null_count} nulls in {col}")
            all_passed = False

    # Check 4: Tenure non-negative
    print("\n[4] Tenure non-negative check...")
    negative_tenure = reviewer_df.filter(F.col("tenure_days") < 0).count()
    if negative_tenure == 0:
        print(f"    PASS: No negative tenure values")
    else:
        print(f"    FAIL: {negative_tenure} negative tenure values")
        all_passed = False

    # Check 5: Rates between 0 and 1
    print("\n[5] Rate range checks (0 to 1)...")
    rate_columns = ["spam_rate", "max_seller_fraction"]
    for col in rate_columns:
        out_of_range = reviewer_df.filter((F.col(col) < 0) | (F.col(col) > 1)).count()
        if out_of_range == 0:
            print(f"    PASS: {col} in valid range")
        else:
            print(f"    FAIL: {out_of_range} values out of range for {col}")
            all_passed = False

    # Check 6: All review_count >= 1
    print("\n[6] Review count minimum check...")
    invalid_counts = reviewer_df.filter(F.col("review_count") < 1).count()
    if invalid_counts == 0:
        print(f"    PASS: All reviewers have review_count >= 1")
    else:
        print(f"    FAIL: {invalid_counts} reviewers with review_count < 1")
        all_passed = False

    # Check 7: Rating in valid range
    print("\n[7] Rating range check...")
    invalid_ratings = reviews_df.filter((F.col("rating") < 1) | (F.col("rating") > 5)).count()
    if invalid_ratings == 0:
        print(f"    PASS: All ratings in range [1, 5]")
    else:
        print(f"    FAIL: {invalid_ratings} ratings out of range")
        all_passed = False

    # Check 8: Date range
    print("\n[8] Date range check...")
    date_range = reviews_df.select(F.min("review_date"), F.max("review_date")).first()
    min_date = str(date_range[0])
    max_date = str(date_range[1])

    if min_date == "2004-10-20":
        print(f"    PASS: Min date = {min_date}")
    else:
        print(f"    FAIL: Min date = {min_date} (expected 2004-10-20)")
        all_passed = False

    if max_date == "2015-01-10":
        print(f"    PASS: Max date = {max_date}")
    else:
        print(f"    FAIL: Max date = {max_date} (expected 2015-01-10)")
        all_passed = False

    # Summary
    print("\n" + "-" * 40)
    if all_passed:
        print("ALL QUALITY CHECKS PASSED")
    else:
        print("SOME QUALITY CHECKS FAILED")
    print("-" * 40)

    return all_passed


def print_teammate_handoff():
    """Print teammate handoff summary from Section I of the spec."""
    print("\n" + "=" * 60)
    print("TEAMMATE HANDOFF SUMMARY")
    print("=" * 60)

    print("""
Dhruv - L2 (FP-Growth Association Rules)
-----------------------------------------
Input: output_csv/reviewer_profiles.csv + output_csv/reviews_enriched.csv
Key columns to discretize:
  - rating: Already discrete (1-5)
  - review_count: Low (1), Medium (2-5), High (6+)
  - tenure_bin: new / moderate / established / veteran
  - avg_review_length: Short (<50), Medium (50-150), Long (>150)
  - max_seller_fraction: Low (<0.5), High (>=0.5)
  - burst_score: Normal (1), Bursty (2+)

Nitish - L3 (DeBERTa Fine-tuning)
----------------------------------
Input: output_csv/reviews_enriched.csv
Text column: review_text (140 rows may be null - drop or impute)
Label column: is_spam (1=spam, 0=legitimate)
Class imbalance: 13.2% spam - use stratified split + weighted loss

Dhruv - L4 (K-Means / DBSCAN Clustering)
-----------------------------------------
Input: output_csv/reviewer_profiles.csv
Features for clustering (all need StandardScaler):
  - review_count, avg_rating, rating_std, avg_review_length
  - avg_word_count, tenure_days, reviews_per_week, unique_sellers
  - max_seller_fraction, avg_days_between_reviews, burst_score, rating_entropy
Recommend: Log-transform right-skewed features before scaling

Disha - L5 (Classification / Anomaly Detection)
-------------------------------------------------
Input: output_csv/reviewer_profiles.csv (reviewer-level) or reviews_enriched.csv (review-level)
Full feature matrix for classification
Label: spam_rate > 0.5 for reviewer-level, is_spam for review-level
Split strategy: 80/20 stratified, consider group-aware split by user_id
""")


def main():
    """Run the complete L1 ETL/OLAP pipeline."""
    start_time = time.time()

    print("\n" + "=" * 60)
    print("L1 ETL/OLAP PIPELINE - YelpZip Fake Review Detection")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output path: {os.path.dirname(DELTA_PATH)}")

    total_steps = 10

    # Step 1: Create Spark session
    print_header(1, total_steps, "SPARK SESSION")
    step_start = time.time()
    spark = create_spark_session()
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 2: Extract raw data
    print_header(2, total_steps, "DATA EXTRACTION")
    step_start = time.time()
    reviews_df, user_map_df, product_map_df = extract_all_data(spark, DATA_PATH)
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 3: Transform - review-level features
    print_header(3, total_steps, "REVIEW-LEVEL TRANSFORMATION")
    step_start = time.time()
    reviews_df = transform_reviews(reviews_df)
    # Cache for reuse
    reviews_df.cache()
    reviews_df.count()  # Materialize cache
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 4: Transform - reviewer-level aggregation
    print_header(4, total_steps, "REVIEWER-LEVEL AGGREGATION")
    step_start = time.time()
    reviewer_df = transform_reviewers(reviews_df)
    reviewer_df.cache()
    reviewer_df.count()  # Materialize cache
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 5: Transform - seller-level aggregation
    print_header(5, total_steps, "SELLER-LEVEL AGGREGATION")
    step_start = time.time()
    seller_df = transform_sellers(reviews_df, reviewer_df)
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 6: Load - save Delta tables + CSV exports
    print_header(6, total_steps, "DATA LOADING (DELTA + CSV)")
    step_start = time.time()
    table_outputs = save_all_tables(
        reviews_df, reviewer_df, seller_df,
        DELTA_PATH, CSV_PATH
    )
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 7: Build OLAP cubes
    print_header(7, total_steps, "OLAP CUBE CONSTRUCTION")
    step_start = time.time()
    cubes = build_olap_cubes(reviews_df, reviewer_df)
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 8: Save OLAP cubes
    print_header(8, total_steps, "SAVE OLAP CUBES")
    step_start = time.time()
    cube_outputs = save_olap_cubes(cubes, DELTA_PATH, CSV_PATH)
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 9: Generate visualizations
    print_header(9, total_steps, "VISUALIZATION GENERATION")
    step_start = time.time()
    plot_files = generate_all_visualizations(
        reviews_df, reviewer_df, cubes, PLOTS_PATH
    )
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Step 10: Quality checks
    print_header(10, total_steps, "QUALITY VALIDATION")
    step_start = time.time()
    all_passed = run_quality_checks(reviews_df, reviewer_df, seller_df)
    print(f"Completed in {time.time() - step_start:.2f}s")

    # Print teammate handoff
    print_teammate_handoff()

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"\nOutput files:")
    print(f"  Delta tables: {DELTA_PATH}/")
    print(f"  CSV exports: {CSV_PATH}/")
    print(f"  Visualizations: {PLOTS_PATH}/")

    # Cleanup
    reviews_df.unpersist()
    reviewer_df.unpersist()
    stop_spark_session()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Reviewer-level aggregation for YelpZip dataset.

Implements Section D of L1_implementation_spec.md.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def compute_basic_aggregations(reviews_df: DataFrame) -> DataFrame:
    """Compute basic reviewer-level aggregations.

    Args:
        reviews_df: Enriched reviews DataFrame.

    Returns:
        DataFrame with basic reviewer aggregations.
    """
    print("  Computing basic aggregations...")

    reviewer_agg = (
        reviews_df.groupBy("user_id")
        .agg(
            F.count("*").alias("review_count"),
            F.sum("is_spam").alias("spam_count"),
            F.mean("rating").alias("avg_rating"),
            F.stddev("rating").alias("rating_std"),
            F.mean("review_length").alias("avg_review_length"),
            F.mean("word_count").alias("avg_word_count"),
            F.min("review_date").alias("first_review_date"),
            F.max("review_date").alias("last_review_date"),
            F.countDistinct("prod_id").alias("unique_sellers"),
        )
    )

    return reviewer_agg


def add_derived_features(reviewer_agg: DataFrame) -> DataFrame:
    """Add derived features from basic aggregations.

    Args:
        reviewer_agg: DataFrame with basic aggregations.

    Returns:
        DataFrame with derived features added.
    """
    print("  Computing derived features...")

    reviewer_agg = (reviewer_agg
        # spam_rate
        .withColumn("spam_rate", F.col("spam_count") / F.col("review_count"))

        # tenure_days (difference between first and last review)
        .withColumn(
            "tenure_days",
            F.datediff(F.col("last_review_date"), F.col("first_review_date"))
        )

        # reviews_per_week (guard for tenure=0)
        .withColumn(
            "reviews_per_week",
            F.when(
                F.col("tenure_days") > 0,
                F.col("review_count") / (F.col("tenure_days") / 7.0)
            ).otherwise(F.col("review_count").cast("double"))  # For single-day reviewers
        )

        # rating_std: NaN for single-review users -> default to 0
        .withColumn("rating_std", F.coalesce(F.col("rating_std"), F.lit(0.0)))
    )

    return reviewer_agg


def compute_max_seller_fraction(
    reviews_df: DataFrame,
    reviewer_agg: DataFrame
) -> DataFrame:
    """Compute max seller fraction (concentration metric).

    Measures what fraction of a reviewer's reviews go to their most-reviewed seller.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_agg: Reviewer aggregations DataFrame.

    Returns:
        Updated reviewer_agg with max_seller_fraction.
    """
    print("  Computing max seller fraction...")

    # Count reviews per (user, seller) pair
    user_seller_counts = (
        reviews_df.groupBy("user_id", "prod_id")
        .agg(F.count("*").alias("seller_review_count"))
    )

    # Max reviews to any single seller per user
    max_seller = (
        user_seller_counts.groupBy("user_id")
        .agg(F.max("seller_review_count").alias("max_seller_reviews"))
    )

    # Join and compute fraction
    reviewer_agg = reviewer_agg.join(max_seller, on="user_id", how="left")
    reviewer_agg = reviewer_agg.withColumn(
        "max_seller_fraction",
        F.col("max_seller_reviews") / F.col("review_count")
    )

    # Drop intermediate column
    reviewer_agg = reviewer_agg.drop("max_seller_reviews")

    return reviewer_agg


def compute_avg_days_between_reviews(
    reviews_df: DataFrame,
    reviewer_agg: DataFrame
) -> DataFrame:
    """Compute average days between consecutive reviews per reviewer.

    Uses window function to compute lag of review dates.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_agg: Reviewer aggregations DataFrame.

    Returns:
        Updated reviewer_agg with avg_days_between_reviews.
    """
    print("  Computing avg days between reviews...")

    # Window ordered by date within each user
    w = Window.partitionBy("user_id").orderBy("review_date")

    # Add previous review date
    reviews_with_prev = reviews_df.withColumn(
        "prev_date",
        F.lag("review_date").over(w)
    )

    # Compute days since previous review
    reviews_with_prev = reviews_with_prev.withColumn(
        "days_since_prev",
        F.datediff(F.col("review_date"), F.col("prev_date"))
    )

    # Average gap per user (excluding first review which has no previous)
    avg_gap = (
        reviews_with_prev
        .filter(F.col("days_since_prev").isNotNull())
        .groupBy("user_id")
        .agg(F.mean("days_since_prev").alias("avg_days_between_reviews"))
    )

    # Join and fill nulls (single-review users)
    reviewer_agg = reviewer_agg.join(avg_gap, on="user_id", how="left")
    reviewer_agg = reviewer_agg.withColumn(
        "avg_days_between_reviews",
        F.coalesce(F.col("avg_days_between_reviews"), F.lit(0.0))
    )

    return reviewer_agg


def compute_burst_score(
    reviews_df: DataFrame,
    reviewer_agg: DataFrame
) -> DataFrame:
    """Compute burst score - max reviews in any 7-day sliding window.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_agg: Reviewer aggregations DataFrame.

    Returns:
        Updated reviewer_agg with burst_score.
    """
    print("  Computing burst score (7-day window)...")

    # Convert date to epoch seconds for range-based window
    reviews_with_epoch = reviews_df.withColumn(
        "review_epoch",
        F.unix_timestamp(F.col("review_date"))
    )

    # 7-day lookback window (in seconds: 7 * 24 * 60 * 60 = 604800)
    w_burst = (
        Window.partitionBy("user_id")
        .orderBy("review_epoch")
        .rangeBetween(-604800, 0)  # 7 days in seconds lookback
    )

    # Count reviews in the 7-day window ending at each review
    burst_counts = reviews_with_epoch.withColumn(
        "window_count",
        F.count("*").over(w_burst)
    )

    # Max window count per user
    burst_score = (
        burst_counts.groupBy("user_id")
        .agg(F.max("window_count").alias("burst_score"))
    )

    # Join and fill nulls
    reviewer_agg = reviewer_agg.join(burst_score, on="user_id", how="left")
    reviewer_agg = reviewer_agg.withColumn(
        "burst_score",
        F.coalesce(F.col("burst_score"), F.lit(1)).cast(IntegerType())
    )

    return reviewer_agg


def compute_rating_entropy(
    reviews_df: DataFrame,
    reviewer_agg: DataFrame
) -> DataFrame:
    """Compute Shannon entropy of rating distribution per reviewer.

    Entropy = -sum(p * log2(p)) for p > 0
    Higher entropy = more diverse ratings
    Lower entropy = concentrated on few rating values

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_agg: Reviewer aggregations DataFrame.

    Returns:
        Updated reviewer_agg with rating_entropy.
    """
    print("  Computing rating entropy...")

    # Pivot to get counts per rating value
    rating_counts = (
        reviews_df.groupBy("user_id")
        .pivot("rating", [1.0, 2.0, 3.0, 4.0, 5.0])
        .count()
        .fillna(0)
    )

    # Rename columns for clarity
    for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
        col_name = str(r)
        rating_counts = rating_counts.withColumnRenamed(col_name, f"r{int(r)}_count")

    # Compute total per user
    rating_counts = rating_counts.withColumn(
        "total",
        F.col("r1_count") + F.col("r2_count") + F.col("r3_count") + F.col("r4_count") + F.col("r5_count")
    )

    # Compute entropy: -sum(p * log2(p)) for p > 0
    # Initialize entropy expression
    entropy_expr = F.lit(0.0)

    for i in range(1, 6):
        p = F.col(f"r{i}_count") / F.col("total")
        # Only add contribution if p > 0 (to avoid log(0))
        entropy_expr = entropy_expr - F.when(
            p > 0,
            p * F.log2(p)
        ).otherwise(0.0)

    rating_counts = rating_counts.withColumn("rating_entropy", entropy_expr)

    # Join with reviewer_agg
    reviewer_agg = reviewer_agg.join(
        rating_counts.select("user_id", "rating_entropy"),
        on="user_id",
        how="left"
    )

    # Fill nulls with 0
    reviewer_agg = reviewer_agg.withColumn(
        "rating_entropy",
        F.coalesce(F.col("rating_entropy"), F.lit(0.0))
    )

    return reviewer_agg


def transform_reviewers(reviews_df: DataFrame) -> DataFrame:
    """Compute all reviewer-level aggregations and features.

    Args:
        reviews_df: Enriched reviews DataFrame from transform_reviews.

    Returns:
        Reviewer profiles DataFrame with all features.
    """
    print("\n" + "=" * 60)
    print("REVIEWER-LEVEL AGGREGATION")
    print("=" * 60)

    # Step 1: Basic aggregations
    reviewer_agg = compute_basic_aggregations(reviews_df)

    # Step 2: Derived features
    reviewer_agg = add_derived_features(reviewer_agg)

    # Step 3: Max seller fraction
    reviewer_agg = compute_max_seller_fraction(reviews_df, reviewer_agg)

    # Step 4: Average days between reviews
    reviewer_agg = compute_avg_days_between_reviews(reviews_df, reviewer_agg)

    # Step 5: Burst score
    reviewer_agg = compute_burst_score(reviews_df, reviewer_agg)

    # Step 6: Rating entropy
    reviewer_agg = compute_rating_entropy(reviews_df, reviewer_agg)

    # Validate output
    reviewer_count = reviewer_agg.count()
    print(f"\nReviewer aggregation complete: {reviewer_count:,} reviewers")

    # Sample statistics
    print("\nReviewer feature statistics:")
    reviewer_agg.select(
        F.mean("review_count").alias("avg_reviews"),
        F.mean("spam_rate").alias("avg_spam_rate"),
        F.mean("tenure_days").alias("avg_tenure_days"),
        F.mean("max_seller_fraction").alias("avg_concentration"),
        F.mean("burst_score").alias("avg_burst"),
        F.mean("rating_entropy").alias("avg_entropy")
    ).show()

    # Print schema
    print("Reviewer profile schema:")
    reviewer_agg.printSchema()

    return reviewer_agg

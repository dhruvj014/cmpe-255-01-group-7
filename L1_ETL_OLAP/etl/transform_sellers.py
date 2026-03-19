"""Seller/Product-level aggregation for YelpZip dataset.

Implements Section E of L1_implementation_spec.md.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_basic_seller_aggregations(reviews_df: DataFrame) -> DataFrame:
    """Compute basic seller-level aggregations.

    Args:
        reviews_df: Enriched reviews DataFrame.

    Returns:
        DataFrame with basic seller aggregations.
    """
    print("  Computing basic seller aggregations...")

    seller_agg = (
        reviews_df.groupBy("prod_id")
        .agg(
            F.count("*").alias("total_reviews"),
            F.sum("is_spam").alias("spam_reviews"),
            F.countDistinct("user_id").alias("unique_reviewers"),
            F.mean("rating").alias("avg_rating"),
            F.stddev("rating").alias("rating_std"),
            F.min("review_date").alias("first_review_date"),
            F.max("review_date").alias("last_review_date"),
            F.first("business_name").alias("business_name"),  # Keep business name
        )
    )

    return seller_agg


def add_derived_seller_features(seller_agg: DataFrame) -> DataFrame:
    """Add derived features for sellers.

    Args:
        seller_agg: DataFrame with basic seller aggregations.

    Returns:
        DataFrame with derived features added.
    """
    print("  Computing derived seller features...")

    seller_agg = (seller_agg
        # Spam rate
        .withColumn(
            "spam_rate",
            F.col("spam_reviews") / F.col("total_reviews")
        )

        # Rating std: NaN for single-review sellers -> default to 0
        .withColumn(
            "rating_std",
            F.coalesce(F.col("rating_std"), F.lit(0.0))
        )

        # Active days (days between first and last review)
        .withColumn(
            "active_days",
            F.datediff(F.col("last_review_date"), F.col("first_review_date"))
        )

        # Review velocity (reviews per week)
        .withColumn(
            "review_velocity",
            F.when(
                F.col("active_days") > 0,
                F.col("total_reviews") / (F.col("active_days") / 7.0)
            ).otherwise(F.col("total_reviews").cast("double"))
        )
    )

    return seller_agg


def compute_suspicious_reviewer_fraction(
    reviews_df: DataFrame,
    reviewer_agg: DataFrame,
    seller_agg: DataFrame
) -> DataFrame:
    """Compute fraction of suspicious reviewers for each seller.

    A reviewer is considered suspicious if their spam_rate > 0.5

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_agg: Reviewer aggregations with spam_rate.
        seller_agg: Seller aggregations DataFrame.

    Returns:
        Updated seller_agg with suspicious_reviewer_fraction.
    """
    print("  Computing suspicious reviewer fraction...")

    # Identify suspicious reviewers (spam_rate > 0.5)
    suspicious_reviewers = (
        reviewer_agg
        .filter(F.col("spam_rate") > 0.5)
        .select("user_id")
        .withColumn("is_suspicious", F.lit(1))
    )

    # Join with reviews to mark suspicious reviews
    reviews_with_suspicious = (
        reviews_df
        .join(suspicious_reviewers, on="user_id", how="left")
        .fillna(0, subset=["is_suspicious"])
    )

    # Count suspicious reviewers per seller
    suspicious_per_seller = (
        reviews_with_suspicious.groupBy("prod_id")
        .agg(
            F.countDistinct(
                F.when(F.col("is_suspicious") == 1, F.col("user_id"))
            ).alias("suspicious_reviewer_count")
        )
    )

    # Join with seller_agg
    seller_agg = seller_agg.join(suspicious_per_seller, on="prod_id", how="left")

    # Compute fraction
    seller_agg = seller_agg.withColumn(
        "suspicious_reviewer_fraction",
        F.coalesce(
            F.col("suspicious_reviewer_count") / F.col("unique_reviewers"),
            F.lit(0.0)
        )
    )

    # Drop intermediate column
    seller_agg = seller_agg.drop("suspicious_reviewer_count")

    return seller_agg


def transform_sellers(
    reviews_df: DataFrame,
    reviewer_agg: DataFrame
) -> DataFrame:
    """Compute all seller-level aggregations and features.

    Args:
        reviews_df: Enriched reviews DataFrame from transform_reviews.
        reviewer_agg: Reviewer profiles from transform_reviewers.

    Returns:
        Seller profiles DataFrame with all features.
    """
    print("\n" + "=" * 60)
    print("SELLER-LEVEL AGGREGATION")
    print("=" * 60)

    # Step 1: Basic aggregations
    seller_agg = compute_basic_seller_aggregations(reviews_df)

    # Step 2: Derived features
    seller_agg = add_derived_seller_features(seller_agg)

    # Step 3: Suspicious reviewer fraction
    seller_agg = compute_suspicious_reviewer_fraction(
        reviews_df, reviewer_agg, seller_agg
    )

    # Validate output
    seller_count = seller_agg.count()
    print(f"\nSeller aggregation complete: {seller_count:,} sellers")

    # Sample statistics
    print("\nSeller feature statistics:")
    seller_agg.select(
        F.mean("total_reviews").alias("avg_reviews"),
        F.mean("spam_rate").alias("avg_spam_rate"),
        F.mean("unique_reviewers").alias("avg_reviewers"),
        F.mean("review_velocity").alias("avg_velocity"),
        F.mean("suspicious_reviewer_fraction").alias("avg_suspicious_frac")
    ).show()

    # Print schema
    print("Seller profile schema:")
    seller_agg.printSchema()

    return seller_agg

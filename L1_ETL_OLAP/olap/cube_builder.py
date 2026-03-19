"""OLAP cube construction for YelpZip dataset.

Implements Section F of L1_implementation_spec.md.
"""
from typing import Dict
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def add_tenure_bin(reviews_df: DataFrame, reviewer_df: DataFrame) -> DataFrame:
    """Add tenure bin column to reviews by joining reviewer tenure.

    Tenure bins:
        - new: < 30 days
        - moderate: 30-89 days
        - established: 90-364 days
        - veteran: >= 365 days

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles with tenure_days.

    Returns:
        Reviews DataFrame with tenure_bin column.
    """
    # Join tenure_days from reviewer profiles
    reviews_with_tenure = reviews_df.join(
        reviewer_df.select("user_id", "tenure_days"),
        on="user_id",
        how="left"
    )

    # Create tenure bins
    reviews_with_tenure = reviews_with_tenure.withColumn(
        "tenure_bin",
        F.when(F.col("tenure_days") < 30, "new")
         .when(F.col("tenure_days") < 90, "moderate")
         .when(F.col("tenure_days") < 365, "established")
         .otherwise("veteran")
    )

    return reviews_with_tenure


def add_reviewer_features_for_cubes(
    reviews_df: DataFrame,
    reviewer_df: DataFrame
) -> DataFrame:
    """Join reviewer-level features needed for OLAP measures.

    Args:
        reviews_df: Reviews DataFrame (potentially with tenure already).
        reviewer_df: Reviewer profiles.

    Returns:
        Reviews DataFrame with reviewer features joined.
    """
    # Join reviews_per_week and max_seller_fraction
    reviews_cube_ready = reviews_df.join(
        reviewer_df.select("user_id", "reviews_per_week", "max_seller_fraction"),
        on="user_id",
        how="left"
    )

    return reviews_cube_ready


def get_olap_measures():
    """Get standard OLAP measures for cube aggregation.

    Returns:
        List of aggregation expressions.
    """
    return [
        F.count("*").alias("total_reviews"),
        F.sum("is_spam").alias("spam_count"),
        (F.sum("is_spam") / F.count("*")).alias("spam_rate"),
        F.mean("review_length").alias("avg_review_length"),
        F.mean("reviews_per_week").alias("avg_reviewer_velocity"),
        F.mean("max_seller_fraction").alias("avg_max_seller_fraction"),
    ]


def build_time_x_rating_cube(reviews_df: DataFrame) -> DataFrame:
    """Build Time x Rating OLAP cube.

    Args:
        reviews_df: Cube-ready reviews DataFrame.

    Returns:
        DataFrame with aggregations by year_month and rating.
    """
    cube = (
        reviews_df
        .groupBy("year_month", F.col("rating").cast(IntegerType()).alias("rating_int"))
        .agg(*get_olap_measures())
        .orderBy("year_month", "rating_int")
    )

    return cube


def build_tenure_x_rating_cube(reviews_df: DataFrame) -> DataFrame:
    """Build Tenure x Rating OLAP cube.

    Args:
        reviews_df: Cube-ready reviews DataFrame with tenure_bin.

    Returns:
        DataFrame with aggregations by tenure_bin and rating.
    """
    cube = (
        reviews_df
        .groupBy("tenure_bin", F.col("rating").cast(IntegerType()).alias("rating_int"))
        .agg(*get_olap_measures())
        .orderBy("tenure_bin", "rating_int")
    )

    return cube


def build_time_x_tenure_cube(reviews_df: DataFrame) -> DataFrame:
    """Build Time x Tenure OLAP cube.

    Args:
        reviews_df: Cube-ready reviews DataFrame with tenure_bin.

    Returns:
        DataFrame with aggregations by year_month and tenure_bin.
    """
    cube = (
        reviews_df
        .groupBy("year_month", "tenure_bin")
        .agg(*get_olap_measures())
        .orderBy("year_month", "tenure_bin")
    )

    return cube


def build_full_3d_cube(reviews_df: DataFrame) -> DataFrame:
    """Build full 3D OLAP cube (Time x Rating x Tenure).

    Args:
        reviews_df: Cube-ready reviews DataFrame with tenure_bin.

    Returns:
        DataFrame with aggregations by all three dimensions.
    """
    cube = (
        reviews_df
        .groupBy(
            "year_month",
            F.col("rating").cast(IntegerType()).alias("rating_int"),
            "tenure_bin"
        )
        .agg(*get_olap_measures())
        .orderBy("year_month", "rating_int", "tenure_bin")
    )

    return cube


def build_olap_cubes(
    reviews_df: DataFrame,
    reviewer_df: DataFrame
) -> Dict[str, DataFrame]:
    """Build all OLAP cubes.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles DataFrame.

    Returns:
        Dictionary mapping cube names to DataFrames.
    """
    print("\n" + "=" * 60)
    print("OLAP CUBE CONSTRUCTION")
    print("=" * 60)

    # Prepare reviews with tenure bins and reviewer features
    print("  Preparing cube-ready DataFrame...")
    reviews_with_tenure = add_tenure_bin(reviews_df, reviewer_df)
    reviews_cube_ready = add_reviewer_features_for_cubes(reviews_with_tenure, reviewer_df)

    # Cache for repeated use
    reviews_cube_ready.cache()
    print(f"  Cube-ready reviews: {reviews_cube_ready.count():,} rows")

    cubes = {}

    # Cube 1: Time x Rating
    print("\n  Building Time x Rating cube...")
    cubes["time_x_rating"] = build_time_x_rating_cube(reviews_cube_ready)
    print(f"    Cells: {cubes['time_x_rating'].count():,}")

    # Cube 2: Tenure x Rating
    print("  Building Tenure x Rating cube...")
    cubes["tenure_x_rating"] = build_tenure_x_rating_cube(reviews_cube_ready)
    print(f"    Cells: {cubes['tenure_x_rating'].count():,}")

    # Cube 3: Time x Tenure
    print("  Building Time x Tenure cube...")
    cubes["time_x_tenure"] = build_time_x_tenure_cube(reviews_cube_ready)
    print(f"    Cells: {cubes['time_x_tenure'].count():,}")

    # Cube 4: Full 3D
    print("  Building Full 3D cube (Time x Rating x Tenure)...")
    cubes["full_3d"] = build_full_3d_cube(reviews_cube_ready)
    print(f"    Cells: {cubes['full_3d'].count():,}")

    # Unpersist cached data
    reviews_cube_ready.unpersist()

    print("\nOLAP cube construction complete!")

    # Show sample from each cube
    for name, cube in cubes.items():
        print(f"\n{name} sample:")
        cube.show(5, truncate=False)

    return cubes

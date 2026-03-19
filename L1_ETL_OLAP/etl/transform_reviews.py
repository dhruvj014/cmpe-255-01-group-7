"""Review-level feature engineering for YelpZip dataset.

Implements Section C of L1_implementation_spec.md.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_spam_flag(df: DataFrame) -> DataFrame:
    """Add binary spam flag from label column.

    Label: -1=spam, +1=legitimate
    is_spam: 1=spam, 0=legitimate

    Args:
        df: DataFrame with 'label' column.

    Returns:
        DataFrame with 'is_spam' column added.
    """
    return df.withColumn(
        "is_spam",
        F.when(F.col("label") == -1, 1).otherwise(0)
    )


def add_date_features(df: DataFrame) -> DataFrame:
    """Add parsed date and time dimension columns.

    Args:
        df: DataFrame with 'date' column (string format YYYY-MM-DD).

    Returns:
        DataFrame with date features added.
    """
    return (df
        .withColumn("review_date", F.to_date(F.col("date"), "yyyy-MM-dd"))
        .withColumn("year", F.year("review_date"))
        .withColumn("month", F.month("review_date"))
        .withColumn("week_number", F.weekofyear("review_date"))
        .withColumn("day_of_week", F.dayofweek("review_date"))  # 1=Sunday, 7=Saturday
        .withColumn("year_month", F.date_format("review_date", "yyyy-MM"))
    )


def add_text_features(df: DataFrame) -> DataFrame:
    """Add text-based features from review content.

    Features:
        - review_length: Character count
        - word_count: Word count
        - exclamation_count: Number of '!' characters
        - question_count: Number of '?' characters
        - capital_ratio: Uppercase chars / total chars
        - avg_word_length: Average word length

    Args:
        df: DataFrame with 'review_text' column.

    Returns:
        DataFrame with text features added.
    """
    # Review length (character count)
    df = df.withColumn(
        "review_length",
        F.when(
            F.col("review_text").isNotNull(),
            F.length(F.col("review_text"))
        ).otherwise(0)
    )

    # Word count
    df = df.withColumn(
        "word_count",
        F.when(
            F.col("review_text").isNotNull(),
            F.size(F.split(F.col("review_text"), "\\s+"))
        ).otherwise(0)
    )

    # Exclamation count
    df = df.withColumn(
        "exclamation_count",
        F.when(
            F.col("review_text").isNotNull(),
            F.length(F.col("review_text")) - F.length(F.regexp_replace(F.col("review_text"), "!", ""))
        ).otherwise(0)
    )

    # Question count
    df = df.withColumn(
        "question_count",
        F.when(
            F.col("review_text").isNotNull(),
            F.length(F.col("review_text")) - F.length(F.regexp_replace(F.col("review_text"), "\\?", ""))
        ).otherwise(0)
    )

    # Capital ratio (uppercase chars / total chars)
    df = df.withColumn(
        "capital_ratio",
        F.when(
            (F.col("review_text").isNotNull()) & (F.length(F.col("review_text")) > 0),
            (F.length(F.col("review_text")) - F.length(F.regexp_replace(F.col("review_text"), "[A-Z]", "")))
            / F.length(F.col("review_text"))
        ).otherwise(0.0)
    )

    # Average word length (total chars excluding spaces / word_count)
    df = df.withColumn(
        "avg_word_length",
        F.when(
            (F.col("review_text").isNotNull()) & (F.col("word_count") > 0),
            F.length(F.regexp_replace(F.col("review_text"), "\\s+", "")) / F.col("word_count")
        ).otherwise(0.0)
    )

    return df


def transform_reviews(df: DataFrame) -> DataFrame:
    """Apply all review-level feature transformations.

    Args:
        df: Raw reviews DataFrame from extraction.

    Returns:
        Enriched DataFrame with all review-level features.
    """
    print("\n" + "=" * 60)
    print("REVIEW-LEVEL TRANSFORMATION")
    print("=" * 60)

    initial_count = df.count()
    print(f"  Input rows: {initial_count:,}")

    # Apply transformations in sequence
    print("  Adding spam flag...")
    df = add_spam_flag(df)

    print("  Adding date features...")
    df = add_date_features(df)

    print("  Adding text features...")
    df = add_text_features(df)

    # Validate transformations
    final_count = df.count()
    assert final_count == initial_count, f"Row count changed: {initial_count} -> {final_count}"

    # Check for null dates
    null_dates = df.filter(F.col("review_date").isNull()).count()
    print(f"  Null review_dates: {null_dates}")

    # Sample statistics
    print("\n  Feature statistics (sample):")
    df.select(
        F.mean("review_length").alias("avg_review_length"),
        F.mean("word_count").alias("avg_word_count"),
        F.mean("exclamation_count").alias("avg_exclamations"),
        F.mean("capital_ratio").alias("avg_capital_ratio"),
        F.mean("is_spam").alias("spam_rate")
    ).show()

    print(f"\nReview transformation complete: {final_count:,} rows")

    # Print schema for verification
    print("\nEnriched review schema:")
    df.printSchema()

    return df

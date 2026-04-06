"""Data extraction module for YelpZip dataset.

Handles loading of all raw data files with robust parsing for edge cases.
"""
import csv
import os
import tempfile
from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, StringType
)


# Schema definitions based on L1_implementation_spec.md
METADATA_SCHEMA = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("prod_id", IntegerType(), False),
    StructField("rating", FloatType(), False),
    StructField("label", IntegerType(), False),
    StructField("date", StringType(), False),
])

REVIEW_CONTENT_SCHEMA = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("prod_id", IntegerType(), False),
    StructField("date", StringType(), False),
    StructField("review_text", StringType(), True),
])

REVIEW_GRAPH_SCHEMA = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("prod_id", IntegerType(), False),
    StructField("rating", FloatType(), False),
])

USER_MAPPING_SCHEMA = StructType([
    StructField("original_yelp_id", StringType(), False),
    StructField("numeric_id", IntegerType(), False),
])

PRODUCT_MAPPING_SCHEMA = StructType([
    StructField("business_name", StringType(), False),
    StructField("numeric_id", IntegerType(), False),
])

# Expected row counts for validation
EXPECTED_METADATA_ROWS = 608598
EXPECTED_REVIEWER_COUNT = 260277
EXPECTED_PRODUCT_COUNT = 5044


def load_metadata(spark: SparkSession, data_path: str) -> DataFrame:
    """Load metadata file containing review labels and ratings.

    Args:
        spark: Active Spark session.
        data_path: Path to YelpZip data directory.

    Returns:
        DataFrame with columns: user_id, prod_id, rating, label, date
    """
    file_path = os.path.join(data_path, "metadata")
    print(f"  Loading metadata from: {file_path}")

    meta_df = spark.read.csv(
        file_path,
        sep="\t",
        header=False,
        schema=METADATA_SCHEMA
    )

    row_count = meta_df.count()
    print(f"  Metadata loaded: {row_count:,} rows")

    if row_count != EXPECTED_METADATA_ROWS:
        print(f"  WARNING: Expected {EXPECTED_METADATA_ROWS:,} rows, got {row_count:,}")

    return meta_df


def load_review_content_robust(spark: SparkSession, data_path: str) -> DataFrame:
    """Load review content with robust parsing for embedded tabs.

    Uses Python pre-processing to handle 140 rows with tab characters in review text.
    Splits on first 3 tabs only to preserve embedded tabs in review_text.

    Args:
        spark: Active Spark session.
        data_path: Path to YelpZip data directory.

    Returns:
        DataFrame with columns: user_id, prod_id, date, review_text
    """
    file_path = os.path.join(data_path, "reviewContent")
    print(f"  Loading reviewContent from: {file_path}")
    print("  Using robust parser (split on first 3 tabs only)...")

    rows = []
    error_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t", 3)  # Split on first 3 tabs only

            if len(parts) >= 4:
                try:
                    user_id = int(parts[0])
                    prod_id = int(parts[1])
                    date = parts[2]
                    review_text = parts[3]
                    rows.append((user_id, prod_id, date, review_text))
                except ValueError:
                    error_count += 1
            elif len(parts) == 3:
                # Row with empty review text
                try:
                    user_id = int(parts[0])
                    prod_id = int(parts[1])
                    date = parts[2]
                    rows.append((user_id, prod_id, date, ""))
                except ValueError:
                    error_count += 1
            else:
                error_count += 1

    print(f"  Parsed {len(rows):,} rows successfully, {error_count} errors")

    # Write to a temp CSV so Spark reads natively (JVM-side) instead of going
    # through py4j row-by-row, which crashes Python 3.12 workers during shuffles.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    writer = csv.writer(tmp)
    writer.writerow(["user_id", "prod_id", "date", "review_text"])
    writer.writerows(rows)
    tmp.close()

    content_df = (
        spark.read.csv(tmp.name, header=True, schema=REVIEW_CONTENT_SCHEMA)
    )
    return content_df


def load_review_graph(spark: SparkSession, data_path: str) -> DataFrame:
    """Load review graph (bipartite user-product edges with ratings).

    Args:
        spark: Active Spark session.
        data_path: Path to YelpZip data directory.

    Returns:
        DataFrame with columns: user_id, prod_id, rating
    """
    file_path = os.path.join(data_path, "reviewGraph")
    print(f"  Loading reviewGraph from: {file_path}")

    graph_df = spark.read.csv(
        file_path,
        sep="\t",
        header=False,
        schema=REVIEW_GRAPH_SCHEMA
    )

    row_count = graph_df.count()
    print(f"  ReviewGraph loaded: {row_count:,} rows")

    return graph_df


def load_user_mapping(spark: SparkSession, data_path: str) -> DataFrame:
    """Load user ID mapping (original Yelp IDs to numeric IDs).

    Args:
        spark: Active Spark session.
        data_path: Path to YelpZip data directory.

    Returns:
        DataFrame with columns: original_yelp_id, numeric_id
    """
    file_path = os.path.join(data_path, "userIdMapping")
    print(f"  Loading userIdMapping from: {file_path}")

    user_map_df = spark.read.csv(
        file_path,
        sep="\t",
        header=False,
        schema=USER_MAPPING_SCHEMA
    )

    row_count = user_map_df.count()
    print(f"  UserIdMapping loaded: {row_count:,} rows")

    if row_count != EXPECTED_REVIEWER_COUNT:
        print(f"  WARNING: Expected {EXPECTED_REVIEWER_COUNT:,} rows, got {row_count:,}")

    return user_map_df


def load_product_mapping(spark: SparkSession, data_path: str) -> DataFrame:
    """Load product ID mapping (business names to numeric IDs).

    Args:
        spark: Active Spark session.
        data_path: Path to YelpZip data directory.

    Returns:
        DataFrame with columns: business_name, numeric_id
    """
    file_path = os.path.join(data_path, "productIdMapping")
    print(f"  Loading productIdMapping from: {file_path}")

    product_map_df = spark.read.csv(
        file_path,
        sep="\t",
        header=False,
        schema=PRODUCT_MAPPING_SCHEMA
    )

    row_count = product_map_df.count()
    print(f"  ProductIdMapping loaded: {row_count:,} rows")

    if row_count != EXPECTED_PRODUCT_COUNT:
        print(f"  WARNING: Expected {EXPECTED_PRODUCT_COUNT:,} rows, got {row_count:,}")

    return product_map_df


def join_reviews_master(
    meta_df: DataFrame,
    content_df: DataFrame,
    product_map_df: DataFrame
) -> DataFrame:
    """Join metadata with review content and business names.

    Args:
        meta_df: Metadata DataFrame with labels.
        content_df: Review content DataFrame with text.
        product_map_df: Product mapping DataFrame with business names.

    Returns:
        Master review DataFrame with all columns joined.
    """
    print("  Joining metadata with review content...")

    # Join metadata + reviewContent on (user_id, prod_id, date)
    reviews_df = meta_df.join(
        content_df.select("user_id", "prod_id", "date", "review_text"),
        on=["user_id", "prod_id", "date"],
        how="left"
    )

    # Join business name from product mapping
    print("  Joining business names...")
    reviews_df = reviews_df.join(
        product_map_df.withColumnRenamed("numeric_id", "prod_id"),
        on="prod_id",
        how="left"
    )

    row_count = reviews_df.count()
    null_text_count = reviews_df.filter(reviews_df.review_text.isNull()).count()

    print(f"  Master review table: {row_count:,} rows")
    print(f"  Rows with null review_text: {null_text_count}")

    return reviews_df


def extract_all_data(
    spark: SparkSession,
    data_path: str
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Extract and join all YelpZip data files.

    Args:
        spark: Active Spark session.
        data_path: Path to YelpZip data directory.

    Returns:
        Tuple of (reviews_df, user_mapping_df, product_mapping_df)
    """
    print("\n" + "=" * 60)
    print("EXTRACTION PHASE")
    print("=" * 60)

    # Load all raw files
    meta_df = load_metadata(spark, data_path)
    content_df = load_review_content_robust(spark, data_path)
    product_map_df = load_product_mapping(spark, data_path)
    user_map_df = load_user_mapping(spark, data_path)

    # Join into master review table
    reviews_df = join_reviews_master(meta_df, content_df, product_map_df)

    # Validate key constraints
    print("\n  Validating data integrity...")
    label_values = reviews_df.select("label").distinct().collect()
    label_set = {row.label for row in label_values}
    assert label_set == {-1, 1}, f"Unexpected label values: {label_set}"
    print("  Labels validated: {-1, 1}")

    rating_values = reviews_df.select("rating").distinct().collect()
    rating_set = {row.rating for row in rating_values}
    assert rating_set == {1.0, 2.0, 3.0, 4.0, 5.0}, f"Unexpected rating values: {rating_set}"
    print("  Ratings validated: {1.0, 2.0, 3.0, 4.0, 5.0}")

    print("\nExtraction complete!")
    print(f"  Reviews: {reviews_df.count():,} rows")
    print(f"  Reviewers: {user_map_df.count():,} unique")
    print(f"  Products: {product_map_df.count():,} unique")

    return reviews_df, user_map_df, product_map_df

"""Data loading module for Delta Lake storage and CSV export.

Implements Section G of L1_implementation_spec.md.
"""
import os
import shutil
from typing import Dict, Optional
from pyspark.sql import DataFrame


def save_to_parquet(
    df: DataFrame,
    path: str,
    partition_by: Optional[str] = None,
    mode: str = "overwrite"
) -> None:
    """Save DataFrame as Parquet table.

    Args:
        df: DataFrame to save.
        path: Output path for Parquet table.
        partition_by: Optional column to partition by.
        mode: Write mode (default: overwrite).
    """
    writer = df.write.format("parquet").mode(mode)

    if partition_by:
        writer = writer.partitionBy(partition_by)

    writer.save(path)
    print(f"  Saved Parquet table to: {path}")


def export_to_csv(
    df: DataFrame,
    path: str,
    filename: str
) -> str:
    """Export DataFrame to a single CSV file.

    Uses coalesce(1) to create a single file, then renames it.

    Args:
        df: DataFrame to export.
        path: Output directory for CSV.
        filename: Desired filename (e.g., "reviews.csv").

    Returns:
        Full path to the exported CSV file.
    """
    temp_path = os.path.join(path, f"temp_{filename}")
    final_path = os.path.join(path, filename)

    # Write as single partition
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_path)

    # Find the actual CSV file and rename it
    for f in os.listdir(temp_path):
        if f.startswith("part-") and f.endswith(".csv"):
            shutil.move(
                os.path.join(temp_path, f),
                final_path
            )
            break

    # Clean up temp directory
    shutil.rmtree(temp_path)

    # Get file size
    file_size = os.path.getsize(final_path)
    size_mb = file_size / (1024 * 1024)

    print(f"  Exported CSV: {final_path} ({size_mb:.2f} MB)")
    return final_path


def save_all_tables(
    reviews_df: DataFrame,
    reviewer_df: DataFrame,
    seller_df: DataFrame,
    delta_path: str,
    csv_path: str
) -> Dict[str, str]:
    """Save all enriched tables to Delta and CSV formats.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles DataFrame.
        seller_df: Seller profiles DataFrame.
        delta_path: Base path for Delta tables.
        csv_path: Base path for CSV exports.

    Returns:
        Dictionary mapping table names to output paths.
    """
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    # Ensure directories exist
    os.makedirs(delta_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    outputs = {}

    # Reviews - partitioned by year_month
    print("\nSaving reviews_enriched...")
    reviews_parquet = os.path.join(delta_path, "reviews_enriched")
    save_to_parquet(reviews_df, reviews_parquet, partition_by="year_month")
    reviews_csv = export_to_csv(reviews_df, csv_path, "reviews_enriched.csv")
    outputs["reviews_enriched"] = {"parquet": reviews_parquet, "csv": reviews_csv}

    # Reviewer profiles - no partitioning
    print("\nSaving reviewer_profiles...")
    reviewer_parquet = os.path.join(delta_path, "reviewer_profiles")
    save_to_parquet(reviewer_df, reviewer_parquet)
    reviewer_csv = export_to_csv(reviewer_df, csv_path, "reviewer_profiles.csv")
    outputs["reviewer_profiles"] = {"parquet": reviewer_parquet, "csv": reviewer_csv}

    # Seller profiles - no partitioning
    print("\nSaving seller_profiles...")
    seller_parquet = os.path.join(delta_path, "seller_profiles")
    save_to_parquet(seller_df, seller_parquet)
    seller_csv = export_to_csv(seller_df, csv_path, "seller_profiles.csv")
    outputs["seller_profiles"] = {"parquet": seller_parquet, "csv": seller_csv}

    print("\nAll tables saved successfully!")
    return outputs


def save_olap_cubes(
    cubes: Dict[str, DataFrame],
    delta_path: str,
    csv_path: str
) -> Dict[str, str]:
    """Save all OLAP cubes to Delta and CSV formats.

    Args:
        cubes: Dictionary of cube name -> DataFrame.
        delta_path: Base path for Delta tables.
        csv_path: Base path for CSV exports.

    Returns:
        Dictionary mapping cube names to output paths.
    """
    print("\n" + "=" * 60)
    print("SAVING OLAP CUBES")
    print("=" * 60)

    olap_delta_path = os.path.join(delta_path, "olap_cubes")
    os.makedirs(olap_delta_path, exist_ok=True)

    outputs = {}

    for cube_name, cube_df in cubes.items():
        print(f"\nSaving {cube_name}...")

        # Parquet
        cube_parquet = os.path.join(olap_delta_path, cube_name)
        save_to_parquet(cube_df, cube_parquet)

        # CSV
        cube_csv = export_to_csv(cube_df, csv_path, f"cube_{cube_name}.csv")

        outputs[cube_name] = {"parquet": cube_parquet, "csv": cube_csv}

    print("\nAll OLAP cubes saved successfully!")
    return outputs

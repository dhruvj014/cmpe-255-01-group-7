"""Data loading module for CSV export.

Implements Section G of L1_implementation_spec.md.

Note: Parquet/Delta writes are skipped on Windows because Hadoop's local-FS
committer calls winutils.exe (chmod) which is not available without a full
Hadoop installation. CSV export uses pandas (toPandas → to_csv) to bypass the
Hadoop file-permission layer entirely.
"""
import os
from typing import Dict, Optional
from pyspark.sql import DataFrame


def save_to_parquet(
    df: DataFrame,
    path: str,
    partition_by: Optional[str] = None,
    mode: str = "overwrite"
) -> None:
    """Save DataFrame as Parquet table.

    Skipped on Windows where winutils.exe is absent; logs a warning instead.

    Args:
        df: DataFrame to save.
        path: Output path for Parquet table.
        partition_by: Optional column to partition by.
        mode: Write mode (default: overwrite).
    """
    import platform
    if platform.system() == "Windows":
        print(f"  [SKIP] Parquet write skipped on Windows (no winutils): {path}")
        return

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

    On Windows (no winutils): collects via toPandas() to bypass the Hadoop
    file-committer which requires winutils.exe to set permissions.
    On Linux/Mac: uses the original Spark coalesce(1) writer for efficiency.

    Args:
        df: DataFrame to export.
        path: Output directory for CSV.
        filename: Desired filename (e.g., "reviews.csv").

    Returns:
        Full path to the exported CSV file.
    """
    import platform
    import shutil

    os.makedirs(path, exist_ok=True)
    final_path = os.path.join(path, filename)

    if platform.system() == "Windows":
        pdf = df.toPandas()
        pdf.to_csv(final_path, index=False)
        row_count = len(pdf)
    else:
        temp_path = os.path.join(path, f"temp_{filename}")
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_path)
        for f in os.listdir(temp_path):
            if f.startswith("part-") and f.endswith(".csv"):
                shutil.move(os.path.join(temp_path, f), final_path)
                break
        shutil.rmtree(temp_path)
        row_count = -1  # skip count on non-Windows to avoid extra Spark action

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    row_info = f", {row_count:,} rows" if row_count >= 0 else ""
    print(f"  Exported CSV: {final_path} ({size_mb:.2f} MB{row_info})")
    return final_path


def save_all_tables(
    reviews_df: DataFrame,
    reviewer_df: DataFrame,
    seller_df: DataFrame,
    delta_path: str,
    csv_path: str
) -> Dict[str, str]:
    """Save all enriched tables to CSV (and Parquet where supported).

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles DataFrame.
        seller_df: Seller profiles DataFrame.
        delta_path: Base path for Parquet tables.
        csv_path: Base path for CSV exports.

    Returns:
        Dictionary mapping table names to output paths.
    """
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    os.makedirs(delta_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    outputs = {}

    print("\nSaving reviews_enriched...")
    save_to_parquet(reviews_df, os.path.join(delta_path, "reviews_enriched"), partition_by="year_month")
    reviews_csv = export_to_csv(reviews_df, csv_path, "reviews_enriched.csv")
    outputs["reviews_enriched"] = {"csv": reviews_csv}

    print("\nSaving reviewer_profiles...")
    save_to_parquet(reviewer_df, os.path.join(delta_path, "reviewer_profiles"))
    reviewer_csv = export_to_csv(reviewer_df, csv_path, "reviewer_profiles.csv")
    outputs["reviewer_profiles"] = {"csv": reviewer_csv}

    print("\nSaving seller_profiles...")
    save_to_parquet(seller_df, os.path.join(delta_path, "seller_profiles"))
    seller_csv = export_to_csv(seller_df, csv_path, "seller_profiles.csv")
    outputs["seller_profiles"] = {"csv": seller_csv}

    print("\nAll tables saved successfully!")
    return outputs


def save_olap_cubes(
    cubes: Dict[str, DataFrame],
    delta_path: str,
    csv_path: str
) -> Dict[str, str]:
    """Save all OLAP cubes to CSV (and Parquet where supported).

    Args:
        cubes: Dictionary of cube name -> DataFrame.
        delta_path: Base path for Parquet tables.
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
        save_to_parquet(cube_df, os.path.join(olap_delta_path, cube_name))
        cube_csv = export_to_csv(cube_df, csv_path, f"cube_{cube_name}.csv")
        outputs[cube_name] = {"csv": cube_csv}

    print("\nAll OLAP cubes saved successfully!")
    return outputs

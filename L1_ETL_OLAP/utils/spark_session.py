"""Spark session configuration for local processing."""
import os
from pyspark.sql import SparkSession
from typing import Optional

# JDK 23+ removed SecurityManager; Spark still needs the legacy flag at JVM startup
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--conf spark.driver.extraJavaOptions=-Djava.security.manager=allow "
    "pyspark-shell"
)

# Windows App Execution Aliases shadow the real python with a Store stub.
# Point Spark workers at the actual interpreter so they don't hit the stub.
import sys
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

_spark_session: Optional[SparkSession] = None


def create_spark_session() -> SparkSession:
    """Create and configure a Spark session for YelpZip processing.

    Returns:
        SparkSession: Configured Spark session for YelpZip ETL/OLAP pipeline.
    """
    global _spark_session

    if _spark_session is not None:
        return _spark_session

    _spark_session = (
        SparkSession.builder
        .appName("YelpZip_L1_ETL_OLAP")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )

    # Set log level to reduce noise
    _spark_session.sparkContext.setLogLevel("ERROR")

    print(f"Spark session created: {_spark_session.version}")
    return _spark_session


def stop_spark_session() -> None:
    """Stop the active Spark session and release resources."""
    global _spark_session

    if _spark_session is not None:
        _spark_session.stop()
        _spark_session = None
        print("Spark session stopped.")

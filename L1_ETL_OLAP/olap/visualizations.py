"""Publication-quality visualizations for YelpZip OLAP analysis.

Implements Section H of L1_implementation_spec.md.
"""
import os
from typing import Dict, List
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_heatmap_month_x_rating(
    cube_time_rating: DataFrame,
    output_path: str
) -> str:
    """Create heatmap of spam rate by month and rating.

    Args:
        cube_time_rating: Time x Rating OLAP cube DataFrame.
        output_path: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    print("  Creating heatmap: Month x Rating...")

    df_plot = cube_time_rating.toPandas()
    pivot = df_plot.pivot_table(
        index="year_month",
        columns="rating_int",
        values="spam_rate"
    )

    fig, ax = plt.subplots(figsize=(12, 20))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=False,
        ax=ax,
        vmin=0,
        vmax=0.4,
        cbar_kws={'label': 'Spam Rate'}
    )
    ax.set_title("Spam Rate by Month x Star Rating", fontsize=14, fontweight='bold')
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Year-Month")

    plt.tight_layout()
    filepath = os.path.join(output_path, "heatmap_month_x_rating.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_heatmap_tenure_x_rating(
    cube_tenure_rating: DataFrame,
    output_path: str
) -> str:
    """Create heatmap of spam rate by tenure bin and rating.

    Args:
        cube_tenure_rating: Tenure x Rating OLAP cube DataFrame.
        output_path: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    print("  Creating heatmap: Tenure x Rating...")

    df_plot = cube_tenure_rating.toPandas()
    tenure_order = ["new", "moderate", "established", "veteran"]

    pivot = df_plot.pivot_table(
        index="tenure_bin",
        columns="rating_int",
        values="spam_rate"
    )
    pivot = pivot.reindex(tenure_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        ax=ax,
        cbar_kws={'label': 'Spam Rate'}
    )
    ax.set_title("Spam Rate by Reviewer Tenure x Star Rating", fontsize=14, fontweight='bold')
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Reviewer Tenure")

    plt.tight_layout()
    filepath = os.path.join(output_path, "heatmap_tenure_x_rating.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_monthly_spam_trend(
    reviews_df: DataFrame,
    output_path: str
) -> str:
    """Create line chart of monthly spam rate trend.

    Args:
        reviews_df: Enriched reviews DataFrame.
        output_path: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    print("  Creating line chart: Monthly spam trend...")

    monthly_spam = (
        reviews_df.groupBy("year_month")
        .agg((F.sum("is_spam") / F.count("*")).alias("spam_rate"))
        .orderBy("year_month")
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        range(len(monthly_spam)),
        monthly_spam["spam_rate"],
        color="tomato",
        linewidth=1.5
    )

    # Calculate overall spam rate for reference line
    overall_spam_rate = monthly_spam["spam_rate"].mean()

    ax.axhline(
        y=overall_spam_rate,
        color="gray",
        linestyle="--",
        label=f"Overall {overall_spam_rate:.1%}"
    )

    ax.set_title("Monthly Spam Rate Trend (2004-2015)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Spam Rate")
    ax.legend()

    # Set x-ticks at intervals
    tick_positions = list(range(0, len(monthly_spam), 12))  # Every year
    tick_labels = [monthly_spam.iloc[i]["year_month"] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.tight_layout()
    filepath = os.path.join(output_path, "line_monthly_spam_rate.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_spam_by_tenure(
    reviews_df: DataFrame,
    reviewer_df: DataFrame,
    output_path: str
) -> str:
    """Create bar chart of spam rate by tenure bin.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles DataFrame.
        output_path: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    print("  Creating bar chart: Spam by tenure...")

    # Join tenure to reviews
    reviews_with_tenure = reviews_df.join(
        reviewer_df.select("user_id", "tenure_days"),
        on="user_id",
        how="left"
    )

    reviews_with_tenure = reviews_with_tenure.withColumn(
        "tenure_bin",
        F.when(F.col("tenure_days") < 30, "new")
         .when(F.col("tenure_days") < 90, "moderate")
         .when(F.col("tenure_days") < 365, "established")
         .otherwise("veteran")
    )

    tenure_spam = (
        reviews_with_tenure.groupBy("tenure_bin")
        .agg((F.sum("is_spam") / F.count("*")).alias("spam_rate"))
        .toPandas()
    )

    tenure_order = ["new", "moderate", "established", "veteran"]
    tenure_spam = tenure_spam.set_index("tenure_bin").reindex(tenure_order).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        tenure_spam["tenure_bin"],
        tenure_spam["spam_rate"],
        color="steelblue",
        edgecolor="black"
    )

    # Add value labels on bars
    for bar, rate in zip(bars, tenure_spam["spam_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title("Spam Rate by Reviewer Tenure", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tenure Bin")
    ax.set_ylabel("Spam Rate")
    ax.set_ylim(0, max(tenure_spam["spam_rate"]) * 1.2)

    plt.tight_layout()
    filepath = os.path.join(output_path, "bar_spam_by_tenure.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_rating_distribution(
    reviews_df: DataFrame,
    output_path: str
) -> str:
    """Create grouped bar chart of rating distribution for spam vs legitimate.

    Args:
        reviews_df: Enriched reviews DataFrame.
        output_path: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    print("  Creating grouped bar chart: Rating distribution...")

    rating_by_label = (
        reviews_df.groupBy("is_spam", F.col("rating").cast(IntegerType()).alias("rating_int"))
        .count()
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    ratings = [1, 2, 3, 4, 5]

    legit_data = rating_by_label[rating_by_label["is_spam"] == 0].set_index("rating_int")["count"]
    spam_data = rating_by_label[rating_by_label["is_spam"] == 1].set_index("rating_int")["count"]

    x = np.arange(len(ratings))
    legit_counts = [legit_data.get(r, 0) for r in ratings]
    spam_counts = [spam_data.get(r, 0) for r in ratings]

    bars1 = ax.bar(x - width/2, legit_counts, width, label="Legitimate", color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width/2, spam_counts, width, label="Spam", color="tomato", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(ratings)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution: Spam vs Legitimate", fontsize=14, fontweight='bold')
    ax.legend()

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    filepath = os.path.join(output_path, "grouped_bar_rating_dist.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_reviewer_feature_distributions(
    reviewer_df: DataFrame,
    output_path: str
) -> str:
    """Create histograms of key reviewer features.

    Args:
        reviewer_df: Reviewer profiles DataFrame.
        output_path: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    print("  Creating reviewer feature distributions...")

    df = reviewer_df.toPandas()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Review count (log scale)
    ax = axes[0, 0]
    ax.hist(df["review_count"], bins=50, color="steelblue", edgecolor="black", log=True)
    ax.set_title("Review Count per Reviewer")
    ax.set_xlabel("Review Count")
    ax.set_ylabel("Frequency (log)")

    # Tenure days
    ax = axes[0, 1]
    ax.hist(df["tenure_days"], bins=50, color="steelblue", edgecolor="black")
    ax.set_title("Tenure (Days)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Frequency")

    # Max seller fraction
    ax = axes[0, 2]
    ax.hist(df["max_seller_fraction"], bins=20, color="steelblue", edgecolor="black")
    ax.set_title("Max Seller Fraction (Concentration)")
    ax.set_xlabel("Fraction")
    ax.set_ylabel("Frequency")

    # Spam rate
    ax = axes[1, 0]
    ax.hist(df["spam_rate"], bins=20, color="tomato", edgecolor="black")
    ax.set_title("Spam Rate Distribution")
    ax.set_xlabel("Spam Rate")
    ax.set_ylabel("Frequency")

    # Burst score
    ax = axes[1, 1]
    ax.hist(df["burst_score"], bins=range(1, int(df["burst_score"].max()) + 2), color="steelblue", edgecolor="black")
    ax.set_title("Burst Score (Max reviews in 7-day window)")
    ax.set_xlabel("Burst Score")
    ax.set_ylabel("Frequency")

    # Rating entropy
    ax = axes[1, 2]
    ax.hist(df["rating_entropy"], bins=20, color="steelblue", edgecolor="black")
    ax.set_title("Rating Entropy")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Frequency")

    plt.suptitle("Reviewer Feature Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_path, "reviewer_feature_distributions.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def generate_all_visualizations(
    reviews_df: DataFrame,
    reviewer_df: DataFrame,
    cubes: Dict[str, DataFrame],
    output_path: str
) -> List[str]:
    """Generate all publication-quality visualizations.

    Args:
        reviews_df: Enriched reviews DataFrame.
        reviewer_df: Reviewer profiles DataFrame.
        cubes: Dictionary of OLAP cubes.
        output_path: Directory to save plots.

    Returns:
        List of paths to generated plots.
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)
    generated_plots = []

    # Plot 1: Heatmap - Month x Rating
    path = plot_heatmap_month_x_rating(cubes["time_x_rating"], output_path)
    generated_plots.append(path)

    # Plot 2: Heatmap - Tenure x Rating
    path = plot_heatmap_tenure_x_rating(cubes["tenure_x_rating"], output_path)
    generated_plots.append(path)

    # Plot 3: Line chart - Monthly spam trend
    path = plot_monthly_spam_trend(reviews_df, output_path)
    generated_plots.append(path)

    # Plot 4: Bar chart - Spam by tenure
    path = plot_spam_by_tenure(reviews_df, reviewer_df, output_path)
    generated_plots.append(path)

    # Plot 5: Grouped bar - Rating distribution
    path = plot_rating_distribution(reviews_df, output_path)
    generated_plots.append(path)

    # Plot 6: Reviewer feature distributions
    path = plot_reviewer_feature_distributions(reviewer_df, output_path)
    generated_plots.append(path)

    print(f"\nGenerated {len(generated_plots)} visualizations:")
    for p in generated_plots:
        print(f"  - {p}")

    return generated_plots

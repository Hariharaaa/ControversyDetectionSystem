"""
Text preprocessing module for the ESG Controversy Detection System.

Provides text cleaning, normalisation, company name masking, and class
imbalance reporting for raw article data before model training.

Usage:
    python -m src.preprocessor                # Preprocess synthetic data
    python -m src.preprocessor --input PATH   # Preprocess a specific CSV
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# Text Cleaner
# ============================================================================
class TextCleaner:
    """Clean and normalise raw article text for model consumption.

    Pipeline:
        1. Strip HTML tags
        2. Remove URLs
        3. Remove special characters (keep alphanumeric + basic punctuation)
        4. Lowercase
        5. Normalise whitespace
        6. Truncate to max_tokens (whitespace-based pre-truncation)
    """

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s.,;:!?'\"()\-/%$£€]")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to a single text string.

        Args:
            text: Raw article text, potentially containing HTML, URLs, etc.

        Returns:
            Cleaned and normalised text string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Step 1: Strip HTML tags
        text = self._strip_html(text)

        # Step 2: Remove URLs
        text = self.URL_PATTERN.sub("", text)

        # Step 3: Lowercase
        text = text.lower()

        # Step 4: Remove special characters
        text = self.SPECIAL_CHAR_PATTERN.sub(" ", text)

        # Step 5: Normalise whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        # Step 6: Truncate to max tokens
        text = self._truncate(text)

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a batch of text strings.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned text strings.
        """
        return [self.clean(t) for t in texts]

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags using BeautifulSoup."""
        try:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ")
        except Exception:
            # Fallback regex if BS4 fails
            return re.sub(r"<[^>]+>", " ", text)

    def _truncate(self, text: str) -> str:
        """Truncate text to max_tokens based on whitespace tokenisation."""
        tokens = text.split()
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        return " ".join(tokens)


# ============================================================================
# Company Name Masker
# ============================================================================
class CompanyMasker:
    """Mask company names in article text to prevent shortcut learning.

    Replaces known company names with [COMPANY] token so the model
    learns to classify based on event descriptions rather than
    company-specific associations.
    """

    MASK_TOKEN = "[COMPANY]"

    # Common large-cap companies and their variants
    DEFAULT_COMPANIES: list[str] = [
        # Real companies
        "Apple", "Microsoft", "Google", "Alphabet", "Amazon", "Meta", "Facebook",
        "Tesla", "Netflix", "Nvidia", "Samsung", "Intel", "IBM", "Oracle",
        "Shell", "BP", "ExxonMobil", "Exxon", "Chevron", "TotalEnergies",
        "Nestlé", "Nestle", "Unilever", "Procter & Gamble", "P&G",
        "JPMorgan", "JP Morgan", "Goldman Sachs", "Morgan Stanley", "Citigroup",
        "HSBC", "Deutsche Bank", "UBS", "Credit Suisse", "Barclays",
        "Boeing", "Airbus", "Lockheed Martin", "Raytheon",
        "Johnson & Johnson", "Pfizer", "Moderna", "AstraZeneca",
        "Walmart", "Target", "Costco", "Nike", "Adidas",
        "Toyota", "Volkswagen", "BMW", "Mercedes-Benz", "Ford", "GM",
        "Coca-Cola", "PepsiCo",
        # Synthetic company names from ingestion module
        "Acme Corp", "GlobalTech Industries", "Pacific Energy Holdings",
        "Continental Mining Group", "Atlas Pharmaceuticals", "Pinnacle Financial",
        "Nexus Semiconductor", "Vanguard Chemical Corp", "Sterling Bank Group",
        "Meridian Oil & Gas", "Apex Consumer Goods", "Horizon Telecom",
        "Titan Automotive", "Crestview Capital", "Solaris Energy",
        "OceanView Foods", "Ironclad Manufacturing", "Summit Healthcare",
        "Cascade Logistics", "Zenith Technology", "Nordic Shipping Ltd",
        "Emerald Agriculture", "Quantum Data Systems", "Regency Hotels Group",
        "Citadel Defense Corp",
    ]

    def __init__(self, companies: list[str] | None = None):
        self.companies = companies or self.DEFAULT_COMPANIES
        # Sort by length descending to match longer names first
        self.companies = sorted(self.companies, key=len, reverse=True)
        # Pre-compile pattern for efficiency
        escaped = [re.escape(c) for c in self.companies]
        self.pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    def mask(self, text: str) -> str:
        """Replace company names in text with [COMPANY] token.

        Args:
            text: Input text potentially containing company names.

        Returns:
            Text with company names replaced by [COMPANY].
        """
        if not isinstance(text, str):
            return ""
        return self.pattern.sub(self.MASK_TOKEN, text)

    def mask_batch(self, texts: list[str]) -> list[str]:
        """Mask company names in a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of texts with company names masked.
        """
        return [self.mask(t) for t in texts]


# ============================================================================
# Class Imbalance Reporter
# ============================================================================
class ImbalanceReporter:
    """Analyse and report class distribution across controversy categories.

    Generates distribution statistics and a bar chart visualisation to
    inform decisions about loss weighting and sampling strategies.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or config.REPORTS_DIR

    def report(self, df: pd.DataFrame, label_column: str = "category") -> dict:
        """Generate a class imbalance report.

        Args:
            df: DataFrame containing article data with a category column.
            label_column: Name of the column containing category labels.

        Returns:
            Dictionary with distribution statistics.
        """
        if label_column not in df.columns:
            logger.error("Column '%s' not found in DataFrame.", label_column)
            return {}

        counts = df[label_column].value_counts()
        total = len(df)
        proportions = counts / total

        stats = {
            "total_samples": total,
            "num_classes": len(counts),
            "distribution": counts.to_dict(),
            "proportions": proportions.to_dict(),
            "min_class": counts.idxmin(),
            "min_count": int(counts.min()),
            "max_class": counts.idxmax(),
            "max_count": int(counts.max()),
            "imbalance_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else float("inf"),
        }

        # Log summary
        logger.info("Class Distribution Report:")
        logger.info("  Total samples: %d", stats["total_samples"])
        logger.info("  Number of classes: %d", stats["num_classes"])
        logger.info("  Imbalance ratio (max/min): %.2f", stats["imbalance_ratio"])
        for cat, count in counts.items():
            logger.info("    %s: %d (%.1f%%)", cat, count, proportions[cat] * 100)

        # Generate bar chart
        self._plot_distribution(counts)

        return stats

    def _plot_distribution(self, counts: pd.Series) -> None:
        """Save a bar chart of class distribution."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
            ax.set_xlabel("Number of Articles", fontsize=12)
            ax.set_ylabel("Category", fontsize=12)
            ax.set_title("ESG Controversy Category Distribution", fontsize=14, fontweight="bold")

            # Add count labels on bars
            for i, (count, cat) in enumerate(zip(counts.values, counts.index)):
                ax.text(count + 1, i, str(count), va="center", fontsize=10)

            plt.tight_layout()
            output_path = self.output_dir / "class_distribution.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Class distribution chart saved to %s", output_path)
        except ImportError as exc:
            logger.warning("Could not generate distribution chart: %s", exc)


# ============================================================================
# Preprocessing Pipeline
# ============================================================================
def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    title_column: str = "title",
    mask_companies: bool = True,
    max_tokens: int = 256,
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to an article DataFrame.

    Steps:
        1. Clean text (strip HTML, URLs, special chars, lowercase, truncate)
        2. Clean title
        3. Create combined text field (title + text)
        4. Mask company names (optional)
        5. Generate imbalance report

    Args:
        df: Raw article DataFrame.
        text_column: Column name containing article body text.
        title_column: Column name containing article titles.
        mask_companies: Whether to mask company names.
        max_tokens: Maximum token count for truncation.

    Returns:
        Preprocessed DataFrame with additional columns.
    """
    df = df.copy()
    logger.info("Preprocessing %d articles...", len(df))

    # Initialise components
    cleaner = TextCleaner(max_tokens=max_tokens)
    masker = CompanyMasker()
    reporter = ImbalanceReporter()

    # Step 1: Clean text fields
    logger.info("Step 1/4: Cleaning text...")
    df["clean_text"] = cleaner.clean_batch(df[text_column].fillna("").tolist())
    df["clean_title"] = cleaner.clean_batch(df[title_column].fillna("").tolist())

    # Step 2: Create combined field
    logger.info("Step 2/4: Creating combined text field...")
    df["combined_text"] = df["clean_title"] + " " + df["clean_text"]
    df["combined_text"] = df["combined_text"].str.strip()

    # Step 3: Mask company names
    if mask_companies:
        logger.info("Step 3/4: Masking company names...")
        df["masked_text"] = masker.mask_batch(df["combined_text"].tolist())
    else:
        df["masked_text"] = df["combined_text"]

    # Step 4: Generate imbalance report
    logger.info("Step 4/4: Generating imbalance report...")
    if "category" in df.columns:
        reporter.report(df, label_column="category")

    # Remove empty rows
    initial_len = len(df)
    df = df[df["masked_text"].str.len() > 10].reset_index(drop=True)
    removed = initial_len - len(df)
    if removed > 0:
        logger.info("Removed %d articles with insufficient text.", removed)

    # Save processed data
    output_path = config.DATA_PROCESSED_DIR / "processed_articles.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Preprocessed data saved to %s (%d articles)", output_path, len(df))

    return df


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG Article Preprocessing")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.SYNTHETIC_OUTPUT_PATH),
        help="Path to input CSV file",
    )
    parser.add_argument("--no-mask", action="store_true", help="Disable company name masking")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s. Run ingestion first.", input_path)
        raise SystemExit(1)

    df = pd.read_csv(input_path)
    processed_df = preprocess_dataset(df, mask_companies=not args.no_mask)
    print(f"\nPreprocessing complete: {len(processed_df)} articles")
    print(f"Columns: {list(processed_df.columns)}")
